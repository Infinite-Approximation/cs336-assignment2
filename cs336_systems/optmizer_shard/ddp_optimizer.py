from typing import Any, Iterable, List, Type

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
	def __init__(self, params: Iterable, optimizer_cls: Type[Optimizer], **kwargs: Any):
		self.world_size = dist.get_world_size()
		self.rank = dist.get_rank()
		self._all_params_in_order: List[torch.nn.Parameter] = []
		self._local_param_groups: List[dict[str, Any]] = []
		# 记录optimizer的state占用的内存
		self.occupied_memory = 0
		super().__init__(params, kwargs)
		self.optimizer = optimizer_cls(self._local_param_groups, **kwargs)

	def step(self, closure=None, **kwargs: Any):
		# 更新本地参数
		loss = self.optimizer.step(closure, **kwargs)
		# 同步参数，使得所有参数都是最新的
		for i, param in enumerate(self._all_params_in_order):
			owner = i % self.world_size
			dist.broadcast(param.data, src=owner)
		return loss

	def add_param_group(self, param_group: dict[str, Any]):
		# 让父类对param_group进行预处理，得到的 self.param_groups 的最后一个就是新的param_group
		super().add_param_group(param_group)
		normalized_group = self.param_groups[-1]
		# 取出出了params以外的超参数
		local_param_group = {k: v for k, v in normalized_group.items() if k != 'params'}
		local_params = []
		# 取出属于自己的参数，按照每个参数在全局的顺序来取
		for param in normalized_group['params']:
			global_idx = len(self._all_params_in_order)
			if global_idx % self.world_size == self.rank:
				local_params.append(param)
				self.occupied_memory += param.numel() * param.element_size() * 2
			self._all_params_in_order.append(param)
				
		local_param_group['params'] = local_params
		self._local_param_groups.append(local_param_group)
		
		
