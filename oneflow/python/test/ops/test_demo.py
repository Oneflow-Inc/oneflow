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
import oneflow as flow

# test file names and methods names are starts with `test'


def test_foo(test_case):
    # only one arg required
    # you can use `test_case' like unittest.TestCase instance
    pass


@flow.unittest.num_nodes_required(2)
def test_bar(test_case):
    # default num_nodes_required is 1
    pass
