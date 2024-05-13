from flax import linen as nn
import jax
from .types import Array, DType, PRNGKey, Shape


def variance_scaling_init(scale: int, mode: str, distribution: str):
    """Common choices:
    mode: "fan_in", "fan_out", "fan_avg"
    distribution: "normal", "truncated_normal", ...
    """
    def init_fn(key: PRNGKey, shape: Shape, dtype: DType, in_axis: int, out_axis: int):
        fn = jax.nn.initializers.variance_scaling(scale, mode, distribution, in_axis, out_axis)
        return fn(key, shape, dtype)
