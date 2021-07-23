import unittest
from oneflow.compatible import single_client as flow


@flow.unittest.skip_unless_1n1d()
class TestFunctionConfig(flow.unittest.TestCase):
    def test_default_placement_scope(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_placement_scope(flow.scope.placement("cpu", "0:0"))

        @flow.global_function(function_config=func_config)
        def Foo():
            test_case.assertEqual(
                "cpu", flow.current_scope().device_parallel_desc_symbol.device_tag
            )
            return flow.get_variable(
                "w", (10,), initializer=flow.constant_initializer(1)
            )

        Foo().get()

    def test_config_setter_getter(test_case):
        func_config = flow.FunctionConfig()
        func_config.enable_inplace()
        test_case.assertEqual(func_config.function_desc.enable_inplace, True)

    def test_global_function_desc(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_placement_scope(flow.scope.placement("cpu", "0:0"))

        @flow.global_function(function_config=func_config)
        def Foo():
            test_case.assertEqual(
                flow.current_global_function_desc().IsTrainable(), False
            )
            return flow.get_variable(
                "w", (10,), initializer=flow.constant_initializer(1)
            )

        Foo().get()


if __name__ == "__main__":
    unittest.main()
