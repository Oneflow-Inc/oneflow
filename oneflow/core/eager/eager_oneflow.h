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
#include "oneflow/core/job/cluster_instruction.cfg.h"
#include "oneflow/core/vm/instruction.cfg.h"
#include "oneflow/core/vm/instruction.msg.h"

namespace oneflow {

namespace vm {
namespace cfg {

class InstructionListProto;

}  // namespace cfg
}  // namespace vm

namespace eager {
namespace cfg {

class EagerSymbolList;

}  // namespace cfg
}  // namespace eager

namespace eager {

class EagerOneflow final {
 public:
  Maybe<void> RunLogicalInstruction(
      const std::shared_ptr<vm::InstructionMsgList>& instruction_list,
      const std::shared_ptr<eager::cfg::EagerSymbolList>& eager_symbol_list);

  Maybe<void> RunPhysicalInstruction(
      const std::shared_ptr<const ClusterInstructionProto>& cluster_instruction);

  Maybe<void> RunPhysicalInstruction(
      const std::shared_ptr<vm::InstructionMsgList>& instruction_list,
      const std::shared_ptr<eager::cfg::EagerSymbolList>& eager_symbol_list);
  Maybe<void> RunPhysicalInstruction(
      const std::shared_ptr<vm::InstructionMsgList>& instruction_list,
      const eager::EagerSymbolList& eager_symbol_list);
};

}  // namespace eager
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_EAGER_ONEFLOW_H_
