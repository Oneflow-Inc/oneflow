import numpy as np
import oneflow as flow


def _call_of_clip_fn(values, min_value=None, max_value=None):
    # flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def clip_fn(
        values_def=flow.FixedTensorDef(values.shape, dtype=flow.float),
    ):
        with flow.device_prior_placement("gpu", "0:0"):
            return flow.clip_by_value(values_def, min_value, max_value, name="Clip")

    return clip_fn(values).get().ndarray()


def test_clip_by_value(test_case):
    values = np.random.randint(low=-100, high=100, size=(8, 512, 4)).astype(np.float32)
    np_out = np.clip(values, -50, 50)

    of_out = _call_of_clip_fn(values, -50, 50)
    test_case.assertTrue(np.array_equal(np_out, of_out))
