import oneflow as flow

def AddLossUnderNormalMode():
    flow.losses.add_loss(None)

def test_ApiNotImplementedError(test_case):
    test_case.assertRaises(flow.error.ApiNotImplementedError._class, AddLossUnderNormalMode)
