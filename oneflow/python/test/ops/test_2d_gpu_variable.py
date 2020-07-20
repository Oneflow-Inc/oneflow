import oneflow as flow


def test_2d_gpu_variable(test_case):
    flow.enable_eager_execution()
    flow.config.gpu_device_num(2)
    function_config = flow.FunctionConfig()
    function_config.train.model_update_conf(dict(naive_conf={}))
    function_config.train.primary_lr(0.1)
    device_name = "0:0-1"

    @flow.global_function(function_config)
    def Foo():
        with flow.scope.placement("gpu", device_name):
            w = flow.get_variable(
                "w",
                shape=(10,),
                dtype=flow.float,
                initializer=flow.constant_initializer(0),
            )
            print(w.numpy(0))
        flow.losses.add_loss(w)

    Foo()
    Foo()
