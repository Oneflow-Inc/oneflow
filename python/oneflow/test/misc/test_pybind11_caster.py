import unittest
import oneflow as flow
import oneflow.unittest

@flow.unittest.skip_unless_1n1d()
class TestPybind11Caster(flow.unittest.TestCase):
    def test_optional(test_case):
        test_case.assertEqual(flow._oneflow_internal.test_api.increase_if_not_none(1), 2)
        test_case.assertEqual(flow._oneflow_internal.test_api.increase_if_not_none(None), None)

    def test_maybe(test_case):
        test_case.assertEqual(flow._oneflow_internal.test_api.divide(6, 2), 3)
        with test_case.assertRaises(Exception) as context:
            flow._oneflow_internal.test_api.divide(6, 0)
        test_case.assertTrue('Check failed' in str(context.exception))

    def test_maybe_void(test_case):
        flow._oneflow_internal.test_api.throw_if_zero(1)
        with test_case.assertRaises(Exception) as context:
            flow._oneflow_internal.test_api.throw_if_zero(0)
        test_case.assertTrue('Check failed' in str(context.exception))


if __name__ == "__main__":
    unittest.main()
