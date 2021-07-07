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
from __future__ import absolute_import

import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.job.regularizer_conf_pb2 as regularizer_conf_util
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("regularizers.l1_l2")
def l1_l2_regularizer(
    l1: float = 0.01, l2: float = 0.01
) -> regularizer_conf_util.RegularizerConf:
    """This operator creates a L1 and L2 weight regularizer. 

    Args:
        l1 (float, optional): The L1 regularization coefficient. Defaults to 0.01.
        l2 (float, optional): The L2 regularization coefficient. Defaults to 0.01.

    Returns:
        regularizer_conf_util.RegularizerConf: A regularizer that can be used in other layers or operators.
    
    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def conv2d_l1_l2_Job(x: tp.Numpy.Placeholder((1, 256, 32, 32))
        ) -> tp.Numpy:
            initializer = flow.truncated_normal(0.1)
            regularizer = flow.regularizers.l1_l2(l1=0.001, l2=0.001)
            conv2d = flow.layers.conv2d(
                x,
                filters=128,
                kernel_size=3,
                strides=1,
                padding='SAME',
                kernel_initializer=initializer, 
                kernel_regularizer=regularizer,
                name="Conv2d"
            )
            return conv2d


        x = np.random.randn(1, 256, 32, 32).astype(np.float32)
        out = conv2d_l1_l2_Job(x)
    
    """
    regularizer = regularizer_conf_util.RegularizerConf()
    setattr(regularizer.l1_l2_conf, "l1", l1)
    setattr(regularizer.l1_l2_conf, "l2", l2)
    return regularizer


@oneflow_export("regularizers.l1")
def l1_regularizer(l: float = 0.01) -> regularizer_conf_util.RegularizerConf:
    """This operator creates a L1 weight regularizer. 

    Args:
        l (float, optional): The L1 regularization coefficient. Defaults to 0.01.

    Returns:
        regularizer_conf_util.RegularizerConf: A regularizer that can be used in other layers or operators.
    
    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def conv2d_l1_Job(x: tp.Numpy.Placeholder((1, 256, 32, 32))
        ) -> tp.Numpy:
            initializer = flow.truncated_normal(0.1)
            regularizer = flow.regularizers.l1(l=0.001)
            conv2d = flow.layers.conv2d(
                x,
                filters=128,
                kernel_size=3,
                strides=1,
                padding='SAME',
                kernel_initializer=initializer, 
                kernel_regularizer=regularizer,
                name="Conv2d"
            )
            return conv2d


        x = np.random.randn(1, 256, 32, 32).astype(np.float32)
        out = conv2d_l1_Job(x)
            
    """
    return l1_l2_regularizer(l1=l, l2=0.0)


@oneflow_export("regularizers.l2")
def l2_regularizer(l: float = 0.01) -> regularizer_conf_util.RegularizerConf:
    """This operator creates a L2 weight regularizer. 

    Args:
        l (float, optional): The L2 regularization coefficient. Defaults to 0.01.

    Returns:
        regularizer_conf_util.RegularizerConf: A regularizer that can be used in other layers or operators.

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def conv2d_l2_Job(x: tp.Numpy.Placeholder((1, 256, 32, 32))
        ) -> tp.Numpy:
            initializer = flow.truncated_normal(0.1)
            regularizer = flow.regularizers.l2(l=0.001)
            conv2d = flow.layers.conv2d(
                x,
                filters=128,
                kernel_size=3,
                strides=1,
                padding='SAME',
                kernel_initializer=initializer, 
                kernel_regularizer=regularizer,
                name="Conv2d"
            )
            return conv2d


        x = np.random.randn(1, 256, 32, 32).astype(np.float32)
        out = conv2d_l2_Job(x)

    """
    return l1_l2_regularizer(l1=0.0, l2=l)
