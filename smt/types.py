from typing import Any, Sequence

from flax.linen import partitioning
import jax
import jax.numpy as jnp

Config = Any

Array = jnp.ndarray
PRNGKey = jnp.ndarray
DType = jnp.dtype
Shape = Sequence[int]

Mesh = jax.sharding.Mesh
ScanIn = partitioning.ScanIn

AxisNames = tuple[str, ...]

BATCH = "activation_batch"
LENGTH = "activation_length"
HEAD = "activation_heads"
D_KV = "activation_kv"

