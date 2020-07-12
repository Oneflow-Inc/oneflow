import oneflow as flow


def test_enable_eager_execution(test_case):
    flow.enable_eager_execution()
    test_case.assertEqual(flow.eager_execution_enabled(), True)
    flow.enable_eager_execution(False)
    test_case.assertEqual(flow.eager_execution_enabled(), False)
    flow.enable_eager_execution(True)
    test_case.assertEqual(flow.eager_execution_enabled(), True)
