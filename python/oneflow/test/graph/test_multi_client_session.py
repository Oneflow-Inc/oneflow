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

import os
import unittest

import oneflow
import oneflow as flow
import oneflow.framework.session_context as session_ctx
import oneflow.unittest
from oneflow.framework.multi_client_session import MultiClientSession


@flow.unittest.skip_unless_1n1d()
class TestMultiClientSession(unittest.TestCase):
    def test_case1(self):
        sess = session_ctx.GetDefaultSession()
        self.assertTrue(isinstance(sess, MultiClientSession))
        sess.TryInit()
        self.assertEqual(sess.status, sess.Status.INITED)

    def test_case2(self):
        print("test_case2")
        sess = session_ctx.GetDefaultSession()
        self.assertTrue(isinstance(sess, MultiClientSession))
        sess.TryInit()
        self.assertEqual(sess.status, sess.Status.INITED)


if __name__ == "__main__":
    unittest.main()
