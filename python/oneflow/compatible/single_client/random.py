from oneflow.compatible.single_client.ops.array_ops import (
    generate_random_batch_permutation_indices,
    shuffle,
)
from oneflow.compatible.single_client.ops.random_ops import Bernoulli as bernoulli
from oneflow.compatible.single_client.ops.random_util import (
    api_gen_random_seed as gen_seed,
)
from oneflow.compatible.single_client.ops.user_data_ops import api_coin_flip as CoinFlip
from oneflow.compatible.single_client.ops.user_data_ops import (
    api_coin_flip as coin_flip,
)
