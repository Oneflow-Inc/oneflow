import oneflow as flow

def test_no_grad(test_case):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_placement_scope(flow.device_prior_placement("gpu", "0:0"))
    func_config.train.primary_lr(1e-3)
    func_config.train.model_update_conf(dict(naive_conf={}))

    def Run():
        @flow.function(func_config)
        def reduce_sum_job_fn():
            x = flow.get_variable(
                "var", shape=(2, 5), dtype=flow.float32, initializer=flow.random_uniform_initializer()
            )
            y = flow.math.reduce_any(x)
            flow.losses.add_loss(y)
            return y
        reduce_sum_job_fn()
    
    test_case.assertRaises(Exception, Run)
