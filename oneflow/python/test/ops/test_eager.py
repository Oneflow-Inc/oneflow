import oneflow as flow

def test_constant(test_case):
    # TODO(lixinqi)
    return
    with flow.device("gpu:0"):
        flow.constant(0, dtype=flow.int32)
