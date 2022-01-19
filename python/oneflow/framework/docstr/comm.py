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
import oneflow
from oneflow.framework.docstr.utils import add_docstr

add_docstr(
    oneflow.comm.send,
    """Sends a tensor synchronously.

    Args:
        tensor (Tensor): Tensor to send.
        dst (int): Destination rank.
        send_meta (Bool): Whether to send meta information (default is True)

    """,
)

add_docstr(
    oneflow.comm.recv,
    """Receives a tensor synchronously.
    
    All(send_meta is False) or none of shape, dtype and device should have value.

    Args:
        src (int, optional): Source rank. Will receive from any
            process if unspecified.
        shape (optional): output tensor shape.
        dataType (optional): output tensor data type.
        device (optional): output tensor device.
        out (Tensor, optional): Tensor to fill with received data.
    
    Returns:
        if out is None, return received tensor. otherwise got data from out self without return.
    """,
)
