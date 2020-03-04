import oneflow as flow

def test_default_placement_scope(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_placement_scope(flow.fixed_placement("cpu", "0:0"))
    @flow.function(func_config)
    def Foo():
        test_case.assertEqual("cpu", flow.placement.current_scope().default_device_tag)
        return flow.get_variable("w", (10,), initializer=flow.constant_initializer(1))
    Foo().get()
