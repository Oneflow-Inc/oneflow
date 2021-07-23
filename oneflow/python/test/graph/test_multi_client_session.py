"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import unittest
import os

os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "12139"
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"

import oneflow
import oneflow.experimental as flow
import oneflow.python.framework.session_context as session_ctx
from oneflow.python.framework.multi_client_session import MultiClientSession


class TestMultiClientSession(unittest.TestCase):
    def test_case1(self):
        # print("test_case1")
        self.assertTrue(flow.distributed.is_multi_client())
        # print(f"is_multi_client: {flow.distributed.is_multi_client()}")

        sess = session_ctx.GetDefaultSession()
        # print(f"sess type: {type(sess)}")
        self.assertTrue(isinstance(sess, MultiClientSession))

        sess.TryInit()
        self.assertEqual(sess.status, sess.Status.INITED)

        # sess.TryClose()
        # self.assertEqual(sess.status, sess.Status.CLOSED)

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
