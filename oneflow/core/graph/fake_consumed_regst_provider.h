/*
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
*/
#ifndef ONEFLOW_CORE_GRAPH_FAKE_CONSUMED_REGST_PROVIDER_H_
#define ONEFLOW_CORE_GRAPH_FAKE_CONSUMED_REGST_PROVIDER_H_

namespace oneflow {

// Provide a compute task node with a fake input regst, and its output regst can be inferred using
// SBP + Placement. The fake compute task node can help the task graph of one rank to infer blob
// desc, mainly to ensure that the transport task node has the correct input blob desc.
class FakeConsumedRegstProvider {
 public:
  FakeConsumedRegstProvider() = default;
  virtual ~FakeConsumedRegstProvider() = default;

  virtual void ConsumeFakeRegstsIf() = 0;
  virtual void EraseFakeRegstsIf() = 0;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_FAKE_CONSUMED_REGST_PROVIDER_H_
