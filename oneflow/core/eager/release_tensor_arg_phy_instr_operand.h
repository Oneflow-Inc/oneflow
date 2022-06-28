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
#ifndef ONEFLOW_CORE_EAGER_RELEASE_TENSOR_ARG_PHY_INSTR_OPERAND_H_
#define ONEFLOW_CORE_EAGER_RELEASE_TENSOR_ARG_PHY_INSTR_OPERAND_H_

#include <functional>
#include <memory>
#include "oneflow/core/intrusive/intrusive.h"
#include "oneflow/core/vm/phy_instr_operand.h"
#include "oneflow/core/eager/local_dep_object.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/stream.h"
#include "oneflow/core/vm/stream.h"

namespace oneflow {

namespace vm {

class EagerBlobObject;

class ReleaseTensorArgPhyInstrOperand : public PhyInstrOperand {
 public:
  ReleaseTensorArgPhyInstrOperand(const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object,
                                  const Optional<vm::Stream*>& stream)
      : eager_blob_object_(eager_blob_object), output_dependences_() {
    output_dependences_.push_back(CHECK_JUST(eager_blob_object->compute_local_dep_object()));
    if (stream.has_value()) {
      stream_sequential_dependence_ = CHECK_JUST(stream)->schedule_local_dep_object().get();
    }
  }
  ~ReleaseTensorArgPhyInstrOperand() override = default;

  const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object() const {
    return eager_blob_object_;
  }

  const DependenceVector& input_dependences() const override {
    static thread_local DependenceVector empty{};
    return empty;
  }
  const DependenceVector& output_dependences() const override { return output_dependences_; }

 private:
  std::shared_ptr<vm::EagerBlobObject> eager_blob_object_;
  DependenceVector output_dependences_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_RELEASE_TENSOR_ARG_PHY_INSTR_OPERAND_H_
