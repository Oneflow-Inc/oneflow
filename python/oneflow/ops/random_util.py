import random
import sys
import typing

import oneflow.framework.hob as hob
import oneflow.lib.core.enable_if as enable_if


def api_gen_random_seed(seed: typing.Optional[int] = None):
    api = enable_if.unique([consistent_gen_random_seed, mirrored_gen_random_seed])
    return api(seed)


@enable_if.condition(hob.consistent_view_enabled)
def consistent_gen_random_seed(seed=None):
    if seed is None:
        seed = random.randint(-sys.maxsize, sys.maxsize)
    return (seed, True)


@enable_if.condition(hob.mirrored_view_enabled)
def mirrored_gen_random_seed(seed=None):
    if seed is None:
        seed = -1
        has_seed = False
    else:
        has_seed = True
    return (seed, has_seed)
