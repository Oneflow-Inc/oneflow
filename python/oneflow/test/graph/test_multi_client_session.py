import unittest
import os
import oneflow
import oneflow.experimental as flow
import oneflow.framework.session_context as session_ctx
from oneflow.framework.multi_client_session import MultiClientSession


class TestMultiClientSession(unittest.TestCase):
    def test_case1(self):
        self.assertTrue(flow.distributed.is_multi_client())
        sess = session_ctx.GetDefaultSession()
        self.assertTrue(isinstance(sess, MultiClientSession))
        sess.TryInit()
        self.assertEqual(sess.status, sess.Status.INITED)

    def test_case2(self):
        print("test_case2")
        self.assertTrue(flow.distributed.is_multi_client())
        sess = session_ctx.GetDefaultSession()
        self.assertTrue(isinstance(sess, MultiClientSession))
        sess.TryInit()
        self.assertEqual(sess.status, sess.Status.INITED)
        sess.TryClose()
        self.assertEqual(sess.status, sess.Status.CLOSED)


if __name__ == "__main__":
    unittest.main()
