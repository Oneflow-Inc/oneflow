import oneflow as flow
from oneflow import Tensor
from typing import Union, List
from . import current_device, device_count

def get_rng_state(device: Union[int, str, flow.device] = 'cuda') -> Tensor:
    r"""Returns the random number generator state of the specified GPU as a ByteTensor.

    Args:
        device (flow.device or int, optional): The device to return the RNG state of.
            Default: ``'cuda'`` (i.e., ``flow.device('cuda')``, the current CUDA device).
    """
    # TODO (add lazy initialization mechanism in OneFlow)
    # _lazy_init()
    if isinstance(device, str):
        device = flow.device(device)
    elif isinstance(device, int):
        device = flow.device('cuda', device)
    idx = device.index
    if idx is None:
        idx = current_device()
    default_generator = flow.cuda.default_generators[idx]
    return default_generator.get_state()


def get_rng_state_all() -> List[Tensor]:
    r"""Returns a list of ByteTensor representing the random number states of all devices."""

    results = []
    for i in range(device_count()):
        results.append(get_rng_state(i))
    return results
