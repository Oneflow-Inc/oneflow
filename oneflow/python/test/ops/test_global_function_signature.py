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
import unittest
import oneflow as flow
import oneflow.typing as oft
import numpy as np
from typing import Tuple, Dict, List


@flow.unittest.skip_unless_1n1d()
class TestGlobalFunctionSignature(flow.unittest.TestCase):
    def test_annotation_return_None(test_case):
        flow.config.gpu_device_num(1)

        @flow.global_function()
        def foo(x: oft.Numpy.Placeholder((10,))) -> None:
            pass

        data = np.ones((10,), dtype=np.float32)
        test_case.assertTrue(foo(data) is None)

    def test_annotation_Numpy(test_case):
        flow.config.gpu_device_num(1)

        @flow.global_function()
        def foo(x: oft.Numpy.Placeholder((10,))) -> oft.Numpy:
            return x

        data = np.ones((10,), dtype=np.float32)
        test_case.assertTrue(np.array_equal(foo(data), data))

    def test_annotation_ListNumpy(test_case):
        flow.config.gpu_device_num(1)

        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def foo(x: oft.ListNumpy.Placeholder((10,))) -> oft.ListNumpy:
            return x

        data = np.ones((10,), dtype=np.float32)
        test_case.assertTrue(np.array_equal(foo([data])[0], data))

    def test_annotation_ListListNumpy(test_case):
        flow.config.gpu_device_num(1)

        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def foo(x: oft.ListListNumpy.Placeholder((10,))) -> oft.ListListNumpy:
            return x

        data = np.ones((10,), dtype=np.float32)
        test_case.assertTrue(np.array_equal(foo([[data]])[0][0], data))

    def test_annotation_watch_Numpy(test_case):
        data = np.ones((10,), dtype=np.float32)

        def Watch(x: oft.Numpy):
            test_case.assertTrue(np.array_equal(x, data))

        flow.config.gpu_device_num(1)

        @flow.global_function()
        def foo(x: oft.Numpy.Placeholder((10,))) -> oft.Numpy:
            flow.watch(x, Watch)
            return x

        foo(data)

    def test_annotation_watch_ListNumpy(test_case):
        data = np.ones((10,), dtype=np.float32)

        def Watch(x: oft.ListNumpy):
            test_case.assertTrue(np.array_equal(x[0], data))

        flow.config.gpu_device_num(1)

        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def foo(x: oft.ListNumpy.Placeholder((10,))) -> oft.ListNumpy:
            flow.watch(x, Watch)
            return x

        foo([data])

    def test_annotation_watch_ListListNumpy(test_case):
        # TODO(lixinqi): fixed bugs
        return
        data = np.ones((10,), dtype=np.float32)

        def Watch(x: oft.ListListNumpy):
            test_case.assertTrue(np.array_equal(x[0][0], data))

        flow.config.gpu_device_num(1)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def foo(x: oft.ListListNumpy.Placeholder((10,))) -> oft.ListListNumpy:
            flow.watch(x, Watch)
            return x

        foo([[data]])

    def test_annotation_Dict_Numpy(test_case):
        flow.config.gpu_device_num(1)

        @flow.global_function()
        def foo(x: oft.Numpy.Placeholder((10,))) -> Dict[str, oft.Numpy]:
            return {"x": x}

        data = np.ones((10,), dtype=np.float32)
        test_case.assertTrue(np.array_equal(foo(data)["x"], data))

    def test_annotation_Dict_ListNumpy(test_case):
        flow.config.gpu_device_num(1)

        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def foo(x: oft.ListNumpy.Placeholder((10,))) -> Dict[str, oft.ListNumpy]:
            return {"x": x}

        data = np.ones((10,), dtype=np.float32)
        test_case.assertTrue(np.array_equal(foo([data])["x"][0], data))

    def test_annotation_Dict_ListListNumpy(test_case):
        flow.config.gpu_device_num(1)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def foo(
            x: oft.ListListNumpy.Placeholder((10,))
        ) -> Dict[str, oft.ListListNumpy]:
            return {"x": x}

        data = np.ones((10,), dtype=np.float32)
        test_case.assertTrue(np.array_equal(foo([[data]])["x"][0][0], data))

    def test_annotation_Dict_Nesting_Numpy(test_case):
        flow.config.gpu_device_num(1)

        @flow.global_function()
        def foo(x: oft.Numpy.Placeholder((10,))) -> Dict[str, Dict[str, oft.Numpy]]:
            return {"x": {"x": x}}

        data = np.ones((10,), dtype=np.float32)
        test_case.assertTrue(np.array_equal(foo(data)["x"]["x"], data))

    def test_annotation_Dict_Nesting_ListNumpy(test_case):
        flow.config.gpu_device_num(1)

        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def foo(
            x: oft.ListNumpy.Placeholder((10,))
        ) -> Dict[str, Dict[str, oft.ListNumpy]]:
            return {"x": {"x": x}}

        data = np.ones((10,), dtype=np.float32)
        test_case.assertTrue(np.array_equal(foo([data])["x"]["x"][0], data))

    def test_annotation_Dict_Nesting_ListListNumpy(test_case):
        flow.config.gpu_device_num(1)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def foo(
            x: oft.ListListNumpy.Placeholder((10,))
        ) -> Dict[str, Dict[str, oft.ListListNumpy]]:
            return {"x": {"x": x}}

        data = np.ones((10,), dtype=np.float32)
        test_case.assertTrue(np.array_equal(foo([[data]])["x"]["x"][0][0], data))

    def test_annotation_Tuple_Numpy(test_case):
        flow.config.gpu_device_num(1)

        @flow.global_function()
        def foo(x: Tuple[oft.Numpy.Placeholder((10,))]) -> Tuple[oft.Numpy]:
            return x

        data = np.ones((10,), dtype=np.float32)
        test_case.assertTrue(np.array_equal(foo((data,))[0], data))

    def test_annotation_Tuple_ListNumpy(test_case):
        flow.config.gpu_device_num(1)

        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def foo(x: Tuple[oft.ListNumpy.Placeholder((10,))]) -> Tuple[oft.ListNumpy]:
            return x

        data = np.ones((10,), dtype=np.float32)
        test_case.assertTrue(np.array_equal(foo(([data],))[0][0], data))

    def test_annotation_Tuple_ListListNumpy(test_case):
        flow.config.gpu_device_num(1)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def foo(
            x: Tuple[oft.ListListNumpy.Placeholder((10,))]
        ) -> Tuple[oft.ListListNumpy]:
            return x

        data = np.ones((10,), dtype=np.float32)
        test_case.assertTrue(np.array_equal(foo(([[data]],))[0][0][0], data))

    def test_annotation_Callback_Numpy(test_case):
        data = np.ones((10,), dtype=np.float32)

        def Test(x: oft.Numpy):
            test_case.assertTrue(np.array_equal(x, data))

        flow.config.gpu_device_num(1)

        @flow.global_function()
        def foo(x: oft.Numpy.Placeholder((10,))) -> oft.Callback[oft.Numpy]:
            return x

        foo(data)(Test)

    def test_annotation_Callback_ListNumpy(test_case):
        data = np.ones((10,), dtype=np.float32)

        def Test(x: oft.ListNumpy):
            test_case.assertTrue(np.array_equal(x[0], data))

        flow.config.gpu_device_num(1)

        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def foo(x: oft.ListNumpy.Placeholder((10,))) -> oft.Callback[oft.ListNumpy]:
            return x

        foo([data])(Test)

    def test_annotation_Callback_ListListNumpy(test_case):
        data = np.ones((10,), dtype=np.float32)

        def Test(x: oft.ListListNumpy):
            test_case.assertTrue(np.array_equal(x[0][0], data))

        flow.config.gpu_device_num(1)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def foo(
            x: oft.ListListNumpy.Placeholder((10,))
        ) -> oft.Callback[oft.ListListNumpy]:
            return x

        foo([[data]])(Test)

    def test_annotation_Callback_Tuple_Numpy(test_case):
        data = np.ones((10,), dtype=np.float32)

        def Test(x: Tuple[oft.Numpy]):
            test_case.assertTrue(np.array_equal(x[0], data))

        flow.config.gpu_device_num(1)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def foo(x: oft.Numpy.Placeholder((10,))) -> oft.Callback[Tuple[oft.Numpy]]:
            return (x,)

        foo(data)(Test)

    def test_annotation_Callback_Tuple_ListNumpy(test_case):
        data = np.ones((10,), dtype=np.float32)

        def Test(x: Tuple[oft.ListNumpy]):
            test_case.assertTrue(np.array_equal(x[0][0], data))

        flow.config.gpu_device_num(1)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def foo(
            x: oft.ListNumpy.Placeholder((10,))
        ) -> oft.Callback[Tuple[oft.ListNumpy]]:
            return (x,)

        foo([data])(Test)

    def test_annotation_Callback_Tuple_ListListNumpy(test_case):
        data = np.ones((10,), dtype=np.float32)

        def Test(x: Tuple[oft.ListListNumpy]):
            test_case.assertTrue(np.array_equal(x[0][0][0], data))

        flow.config.gpu_device_num(1)

        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def foo(
            x: oft.ListListNumpy.Placeholder((10,))
        ) -> oft.Callback[Tuple[oft.ListListNumpy]]:
            return (x,)

        foo([[data]])(Test)

    def test_annotation_Bundle_Numpy(test_case):
        flow.config.gpu_device_num(1)

        @flow.global_function()
        def foo(x: oft.Numpy.Placeholder((10,))) -> oft.Bundle[oft.Numpy]:
            return x

        data = np.ones((10,), dtype=np.float32)
        test_case.assertTrue(np.array_equal(foo(data), data))

    def test_annotation_Bundle_List_Numpy(test_case):
        flow.config.gpu_device_num(1)

        @flow.global_function()
        def foo(x: oft.Numpy.Placeholder((10,))) -> oft.Bundle[oft.Numpy]:
            return [x]

        data = np.ones((10,), dtype=np.float32)
        test_case.assertTrue(np.array_equal(foo(data)[0], data))

    def test_annotation_Bundle_Dict_Numpy(test_case):
        flow.config.gpu_device_num(1)

        @flow.global_function()
        def foo(x: oft.Numpy.Placeholder((10,))) -> oft.Bundle[oft.Numpy]:
            return {"x": x}

        data = np.ones((10,), dtype=np.float32)
        test_case.assertTrue(np.array_equal(foo(data)["x"], data))

    def test_annotation_Bundle_Tuple_Numpy(test_case):
        flow.config.gpu_device_num(1)

        @flow.global_function()
        def foo(x: oft.Numpy.Placeholder((10,))) -> oft.Bundle[oft.Numpy]:
            return (x,)

        data = np.ones((10,), dtype=np.float32)
        test_case.assertTrue(np.array_equal(foo(data)[0], data))

    def test_annotation_Bundle_Mix_Nesting_Numpy(test_case):
        flow.config.gpu_device_num(1)

        @flow.global_function()
        def foo(x: oft.Numpy.Placeholder((10,))) -> oft.Bundle[oft.Numpy]:
            return (x, (x,), [x, x, x], {"x": {256: x}})

        data = np.ones((10,), dtype=np.float32)
        test_case.assertTrue(np.array_equal(foo(data)[0], data))
        test_case.assertTrue(np.array_equal(foo(data)[1][0], data))
        test_case.assertTrue(np.array_equal(foo(data)[2][0], data))
        test_case.assertTrue(np.array_equal(foo(data)[2][1], data))
        test_case.assertTrue(np.array_equal(foo(data)[2][2], data))
        test_case.assertTrue(np.array_equal(foo(data)[3]["x"][256], data))

    def test_annotation_Bundle_ListNumpy(test_case):
        flow.config.gpu_device_num(1)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def foo(x: oft.ListNumpy.Placeholder((10,))) -> oft.Bundle[oft.ListNumpy]:
            return x

        data = np.ones((10,), dtype=np.float32)
        test_case.assertTrue(np.array_equal(foo([data])[0], data))

    def test_annotation_Bundle_List_ListNumpy(test_case):
        flow.config.gpu_device_num(1)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def foo(x: oft.ListNumpy.Placeholder((10,))) -> oft.Bundle[oft.ListNumpy]:
            return [x]

        data = np.ones((10,), dtype=np.float32)
        test_case.assertTrue(np.array_equal(foo([data])[0][0], data))

    def test_annotation_Bundle_Dict_ListNumpy(test_case):
        flow.config.gpu_device_num(1)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def foo(x: oft.ListNumpy.Placeholder((10,))) -> oft.Bundle[oft.ListNumpy]:
            return {"x": x}

        data = np.ones((10,), dtype=np.float32)
        test_case.assertTrue(np.array_equal(foo([data])["x"][0], data))

    def test_annotation_Bundle_Tuple_ListNumpy(test_case):
        flow.config.gpu_device_num(1)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def foo(x: oft.ListNumpy.Placeholder((10,))) -> oft.Bundle[oft.ListNumpy]:
            return (x,)

        data = np.ones((10,), dtype=np.float32)
        test_case.assertTrue(np.array_equal(foo([data])[0][0], data))

    def test_annotation_Bundle_Mix_Nesting_ListNumpy(test_case):
        flow.config.gpu_device_num(1)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def foo(x: oft.ListNumpy.Placeholder((10,))) -> oft.Bundle[oft.ListNumpy]:
            return (x, (x,), [x, x, x], {"x": {256: x}})

        data = np.ones((10,), dtype=np.float32)
        test_case.assertTrue(np.array_equal(foo([data])[0][0], data))
        test_case.assertTrue(np.array_equal(foo([data])[1][0][0], data))
        test_case.assertTrue(np.array_equal(foo([data])[2][0][0], data))
        test_case.assertTrue(np.array_equal(foo([data])[2][1][0], data))
        test_case.assertTrue(np.array_equal(foo([data])[2][2][0], data))
        test_case.assertTrue(np.array_equal(foo([data])[3]["x"][256][0], data))

    def test_annotation_Bundle_ListListNumpy(test_case):
        flow.config.gpu_device_num(1)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def foo(
            x: oft.ListListNumpy.Placeholder((10,))
        ) -> oft.Bundle[oft.ListListNumpy]:
            return x

        data = np.ones((10,), dtype=np.float32)
        test_case.assertTrue(np.array_equal(foo([[data]])[0][0], data))

    def test_annotation_Bundle_List_ListListNumpy(test_case):
        flow.config.gpu_device_num(1)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def foo(
            x: oft.ListListNumpy.Placeholder((10,))
        ) -> oft.Bundle[oft.ListListNumpy]:
            return [x]

        data = np.ones((10,), dtype=np.float32)
        test_case.assertTrue(np.array_equal(foo([[data]])[0][0][0], data))

    def test_annotation_Bundle_Dict_ListListNumpy(test_case):
        flow.config.gpu_device_num(1)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def foo(
            x: oft.ListListNumpy.Placeholder((10,))
        ) -> oft.Bundle[oft.ListListNumpy]:
            return {"x": x}

        data = np.ones((10,), dtype=np.float32)
        test_case.assertTrue(np.array_equal(foo([[data]])["x"][0][0], data))

    def test_annotation_Bundle_Tuple_ListListNumpy(test_case):
        flow.config.gpu_device_num(1)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def foo(
            x: oft.ListListNumpy.Placeholder((10,))
        ) -> oft.Bundle[oft.ListListNumpy]:
            return (x,)

        data = np.ones((10,), dtype=np.float32)
        test_case.assertTrue(np.array_equal(foo([[data]])[0][0][0], data))

    def test_annotation_Bundle_Mix_Nesting_ListListNumpy(test_case):
        flow.config.gpu_device_num(1)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def foo(
            x: oft.ListListNumpy.Placeholder((10,))
        ) -> oft.Bundle[oft.ListListNumpy]:
            return (x, (x,), [x, x, x], {"x": {256: x}})

        data = np.ones((10,), dtype=np.float32)
        test_case.assertTrue(np.array_equal(foo([[data]])[0][0][0], data))
        test_case.assertTrue(np.array_equal(foo([[data]])[1][0][0][0], data))
        test_case.assertTrue(np.array_equal(foo([[data]])[2][0][0][0], data))
        test_case.assertTrue(np.array_equal(foo([[data]])[2][1][0][0], data))
        test_case.assertTrue(np.array_equal(foo([[data]])[2][2][0][0], data))
        test_case.assertTrue(np.array_equal(foo([[data]])[3]["x"][256][0][0], data))

    def test_annotation_return_List_Numpy(test_case):
        data = np.ones((10,), dtype=np.float32)

        flow.clear_default_session()
        flow.config.gpu_device_num(1)

        @flow.global_function()
        def foo(x: oft.Numpy.Placeholder(shape=data.shape)) -> List[oft.Numpy]:
            return [x, x, x]

        x, y, z = foo(data)
        test_case.assertTrue(np.array_equal(x, data))
        test_case.assertTrue(np.array_equal(y, data))
        test_case.assertTrue(np.array_equal(z, data))

    def test_annotation_return_List_ListNumpy(test_case):
        data = np.ones((10,), dtype=np.float32)

        flow.clear_default_session()

        def foo(x: oft.ListNumpy.Placeholder(shape=data.shape)) -> List[oft.ListNumpy]:
            return [x, x]

        x, y = foo([data])
        test_case.assertTrue(np.array_equal(x[0], data))
        test_case.assertTrue(np.array_equal(y[0], data))

    def test_annotation_return_List_ListListNumpy(test_case):
        data = np.ones((10,), dtype=np.float32)

        flow.clear_default_session()

        def foo(
            x: oft.ListListNumpy.Placeholder(shape=data.shape),
        ) -> List[oft.ListListNumpy]:
            return [x, x]

        x, y = foo([[data]])
        test_case.assertTrue(np.array_equal(x[0][0], data))
        test_case.assertTrue(np.array_equal(y[0][0], data))

    def test_annotation_return_List_Nesting_Tuple(test_case):
        x = np.random.rand(5).astype(np.float32)
        y = np.random.rand(10).astype(np.float32)

        flow.clear_default_session()

        def foo(
            x: oft.Numpy.Placeholder(shape=x.shape),
            y: oft.ListNumpy.Placeholder(shape=y.shape),
        ) -> Tuple[List[oft.Numpy], List[oft.ListNumpy]]:
            return ([x, x, x], [y, y])

        x_list, y_list = foo(x, [y])
        test_case.assertTrue(np.array_equal(x_list[0], x))
        test_case.assertTrue(np.array_equal(x_list[1], x))
        test_case.assertTrue(np.array_equal(x_list[2], x))
        test_case.assertTrue(np.array_equal(y_list[0][0], y))
        test_case.assertTrue(np.array_equal(y_list[1][0], y))


if __name__ == "__main__":
    unittest.main()
