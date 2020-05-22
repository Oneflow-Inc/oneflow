import oneflow as flow

def AddLossUnderNormalMode():
    flow.losses.add_loss(None)

def test_ApiAttributeError(test_case):
    # TODO: update this testcase when enable_if ready
    test_case.assertRaises(AttributeError, AddLossUnderNormalMode)
