import unittest
from oneflow.compatible import single_client as flow


@unittest.skipIf(flow.unittest.env.device_num() != 1, "only runs when device_num is 1")
class TestDemo(flow.unittest.TestCase):
    @unittest.skipIf(
        flow.unittest.env.node_size() != 1, "only runs when node_size is 1"
    )
    def test_foo(test_case):
        pass

    @unittest.skipIf(
        flow.unittest.env.node_size() != 2, "only runs when node_size is 2"
    )
    def test_bar(test_case):
        pass


if __name__ == "__main__":
    unittest.main()
