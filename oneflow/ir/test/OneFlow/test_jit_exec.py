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
# RUN: python3 %s
import oneflow as flow


class MyModule(flow.nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        # This parameter will be copied to the new ScriptModule
        self.weight = flow.nn.Parameter(flow.rand(N, M))

        # When this submodule is used, it will be compiled
        self.linear = flow.nn.Linear(N, M)

    @flow.jit.exec
    def forward(self, input):
        output = self.linear(input)
        print(output)
        return output


linear = MyModule(2, 3)
for i in range(100):
    print(linear(flow.randn(2, 2)))
