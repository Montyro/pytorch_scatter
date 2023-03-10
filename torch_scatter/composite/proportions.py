from typing import Optional

import torch

from torch_scatter import scatter_sum, scatter_max
from torch_scatter.utils import broadcast


def scatter_proportions(src: torch.Tensor, index: torch.Tensor,
                    dim: int = -1,
                    dim_size: Optional[int] = None) -> torch.Tensor:
    if not torch.is_floating_point(src):
        raise ValueError('`scatter_proportions` can only be computed over tensors '
                         'with floating point data types.')

    index = broadcast(index, src, dim)

    ##max_value_per_index = scatter_max(
    #    src, index, dim=dim, dim_size=dim_size)[0]
    #max_per_src_element = max_value_per_index.gather(dim, index)

    #recentered_scores = src - max_per_src_element
    #recentered_scores_exp = recentered_scores.exp_()

    sum_per_index = scatter_sum(
        src, index, dim, dim_size=dim_size)
    normalizing_constants = sum_per_index.gather(dim, index)

    return src.div(normalizing_constants)


