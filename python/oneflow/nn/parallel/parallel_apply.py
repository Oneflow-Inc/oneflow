"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import threading
import oneflow as flow
from oneflow.cuda._utils import _get_device_index
from oneflow.cuda.amp import autocast
from oneflow._utils import ExceptionWrapper


def get_a_var(obj):
    if isinstance(obj, flow.Tensor):
        return obj

    if isinstance(obj, list) or isinstance(obj, tuple):
        for result in map(get_a_var, obj):
            if isinstance(result, flow.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, flow.Tensor):
                return result
    return None


def parallel_apply(modules, inputs, kwargs_tup=None, devices=None):
    r"""Applies each `module` in :attr:`modules` in parallel on arguments
    contained in :attr:`inputs` (positional) and :attr:`kwargs_tup` (keyword)
    on each of :attr:`devices`.

    Args:
        modules (Module): modules to be parallelized
        inputs (tensor): inputs to the modules
        devices (list of int or flow.device): CUDA devices

    :attr:`modules`, :attr:`inputs`, :attr:`kwargs_tup` (if given), and
    :attr:`devices` (if given) should all have same length. Moreover, each
    element of :attr:`inputs` can either be a single object as the only argument
    to a module, or a collection of positional arguments.
    """
    assert len(modules) == len(inputs)
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    devices = [_get_device_index(x, True) for x in devices]
    # oneflow not support flow.cuda.current_stream flow.cuda.current_stream(x)
    streams = [None for x in devices]
    lock = threading.Lock()
    results = {}
    grad_enabled, autocast_enabled = flow.is_grad_enabled(), flow.is_autocast_enabled()
    print(grad_enabled, autocast_enabled)

    def _worker(i, module, input, kwargs, device=None, stream=None):
        flow.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        if stream is None:
            # oneflow not support flow.cuda.current_stream
            # stream = flow.cuda.current_stream(device)
            stream = None
        try:
            # flow.cuda.stream(stream)
            # with flow.cuda.device(device), autocast(enabled=autocast_enabled):
            # this also avoids accidental slicing of `input` if it is a Tensor
            if not isinstance(input, (list, tuple)):
                input = (input,)
            output = module(*input, **kwargs)
            with lock:
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(
                    where="in replica {} on device {}".format(i, device)
                )

    if len(modules) > 1:
        threads = [
            threading.Thread(
                target=_worker, args=(i, module, input, kwargs, device, stream)
            )
            for i, (module, input, kwargs, device, stream) in enumerate(
                zip(modules, inputs, kwargs_tup, devices, streams)
            )
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0], streams[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    return outputs
