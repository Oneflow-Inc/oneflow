import oneflow as flow

def AddLossUnderNormalMode():
    flow.losses.add_loss(None)

def test_ApiAttributeError(test_case):
    test_case.assertRaises(AttributeError, AddLossUnderNormalMode)
