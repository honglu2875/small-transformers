#  Copyright 2024 Honglu Fan
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from flax import linen as nn
import jax
from smt.types import Array, DType, PRNGKey, Shape


def variance_scaling_init(scale: int, mode: str, distribution: str):
    """Common choices:
    mode: "fan_in", "fan_out", "fan_avg"
    distribution: "normal", "truncated_normal", ...
    """
    def init_fn(key: PRNGKey, shape: Shape, dtype: DType, in_axis: int, out_axis: int):
        fn = jax.nn.initializers.variance_scaling(scale, mode, distribution, in_axis, out_axis)
        return fn(key, shape, dtype)
