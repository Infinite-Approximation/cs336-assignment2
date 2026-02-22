import torch
import torch.distributed as dist
from typing import Any, List, Optional
from dataclasses import dataclass

class DDP(torch.nn.Module):
	def __init__(self, module: torch.nn.Module):
		super().__init__()
		self.module = module
		self._pending_works = []
		self.world_size = dist.get_world_size()
		for p in self.module.parameters():
			dist.broadcast(p.data, src=0)
		
		# 注册反向传播之后自动执行的hook
		def _post_acc_hook(param: torch.nn.Parameter):
			# 发起异步的all_reduce
			work = dist.all_reduce(param.grad, async_op=True)
			self._pending_works.append((param, work))

		for p in self.module.parameters():
			if p.requires_grad:
				p.register_post_accumulate_grad_hook(_post_acc_hook)


	def forward(self, *inputs: Any, **kwargs: Any):
		return self.module(*inputs, **kwargs)

	def finish_gradient_synchronization(self) -> None:
		for param, work in self._pending_works:
			work.wait()
			param.grad /= self.world_size
		self._pending_works.clear()

@dataclass
class Bucket:
	params: List[torch.nn.Parameter]
	offsets: List[int]
	buffer: torch.Tensor
	param2idx: dict[int, int] # 找到param是bucket中的位置
	ready: int = 0
	work: Optional[dist.Work] = None
	@classmethod
	def from_params(cls, params: List[torch.nn.Parameter]) -> "Bucket":
		offsets, cur = [0], 0
		numels = [p.numel() for p in params]
		for numel in numels:
			cur += numel
			offsets.append(cur)
		buffer = torch.zeros(cur, dtype=params[0].dtype, device=params[0].device)
		param2idx = {id(p): i for i, p in enumerate(params)}
		return cls(params=params, offsets=offsets, buffer=buffer, param2idx=param2idx)

	def add_grad(self, p: torch.nn.Parameter):
		"""
		将对应的已经反向传播好的参数的梯度拷贝到buffer中 
		"""
		idx = self.param2idx[id(p)]
		buffer_start, buffer_end = self.offsets[idx], self.offsets[idx+1]
		self.buffer[buffer_start: buffer_end].copy_(p.grad.view(-1))
		self.ready += 1
		return self.ready == len(self.params)

	def launch(self):
		self.work = dist.all_reduce(self.buffer, async_op=True)
	
	def finalize(self, world_size: int):
		if self.work is None:
			return
		self.work.wait()
		self.buffer.div_(world_size)
		for idx, p in enumerate(self.params):
			buffer_start, buffer_end = self.offsets[idx], self.offsets[idx+1]
			p.grad.copy_(self.buffer[buffer_start: buffer_end].view_as(p.grad))
		
		# 清理状态，防止影响下一个iter
		self.work = None
		self.ready = 0
		

class DDPWithBucket(torch.nn.Module):
	def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
		super().__init__()
		self.module = module
		self.world_size = dist.get_world_size()
		for p in self.module.parameters():
			dist.broadcast(p.data, src=0)
		self.buckets: List[Bucket] = []
		self.param2bucket: dict[int, Bucket] = {}
		self._build_buckets(bucket_size_mb)
		
		# 注册反向传播之后自动执行的hook
		def _post_acc_hook(param: torch.nn.Parameter):
			if param.grad is None:
				return
			# 找出对应的桶
			bucket = self.param2bucket[id(param)]
			if bucket.add_grad(param):
				bucket.launch()

		for p in self.module.parameters():
			if p.requires_grad:
				p.register_post_accumulate_grad_hook(_post_acc_hook)

	def _build_buckets(self, bucket_size_mb: float):
		bucket_size_in_bytes = 1024 * 1024 * bucket_size_mb
		params = [p for p in reversed(list(self.module.parameters()))]
		cur_params, cur_bytes = [], 0
		for p in params:
			if not p.requires_grad:
				continue
			p_bytes = p.numel() * p.element_size()
			# 如果加上当前p会超过bucket size，那就将当前的params当做一个bucket
			if cur_params and cur_bytes + p_bytes > bucket_size_in_bytes:
				bucket = Bucket.from_params(cur_params)
				self.buckets.append(bucket)
				cur_params, cur_bytes = [], 0
			cur_params.append(p)	
			cur_bytes += p_bytes
		# 可能cur_params里面的params没超过bucket_size，但是遍历完了，此时需要添加cur_params
		if cur_params:
			bucket = Bucket.from_params(cur_params)
			self.buckets.append(bucket)

		# 需要添加一个param到bucket的映射，这样才能知道当param完成反向传播后，哪些bucket需要修改
		for b in self.buckets:
			for p in b.params:
				self.param2bucket[id(p)] = b

	
	def forward(self, *inputs: Any, **kwargs: Any):
		return self.module(*inputs, **kwargs)

	def finish_gradient_synchronization(self) -> None:
		for b in self.buckets:
			b.finalize(self.world_size)