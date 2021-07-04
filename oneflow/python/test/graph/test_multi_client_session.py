import unittest

import oneflow as flow

@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestMultiClientSession(flow.unittest.TestCase):
    def test_multi_client_sessioin(test_case):
        import oneflow.python.framework.session_context as sc
        s = sc.GetDefaultSession()
        print("default session id ", s.id)
        print("default session type ", type(s))

if __name__ == "__main__":
    unittest.main()
