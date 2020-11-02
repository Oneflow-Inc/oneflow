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
#ifndef ONEFLOW_CORE_EAGER_EAGER_ONEFLOW_H_
#define ONEFLOW_CORE_EAGER_EAGER_ONEFLOW_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job/cluster_instruction.pb.h"

namespace oneflow {
namespace eager {

class EagerOneflow final {
 public:
  Maybe<void> RunLogicalInstruction(
      const std::shared_ptr<const ClusterInstructionProto>& cluster_instruction);

  Maybe<void> RunLogicalInstruction(const std::string& instruction_list_proto_str,
                                    const std::string& eager_symbol_list_str);

  Maybe<void> RunPhysicalInstruction(const std::string& instruction_list_proto_str,
                                     const std::string& eager_symbol_list_str);
  Maybe<void> RunPhysicalInstruction(
      const std::shared_ptr<const ClusterInstructionProto>& cluster_instruction);
};

}  // namespace eager
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_EAGER_ONEFLOW_H_
