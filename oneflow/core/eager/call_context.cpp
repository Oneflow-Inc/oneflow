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
#include "oneflow/core/eager/call_context.h"
#include "oneflow/core/eager/tensor_storage.h"

namespace oneflow {
namespace eager {
namespace {

vm::WeakEagerBlobObjectList shared_to_weak(const vm::EagerBlobObjectList& shared_list) {
  vm::WeakEagerBlobObjectList ret;
  ret.reserve(shared_list.size());
  for (const auto& shared : shared_list) { ret.emplace_back(shared); }
  return ret;
}

}  // namespace
DtrCallContext::DtrCallContext(const CallContext& call_ctx)
    : composed_attrs_(call_ctx.composed_attrs()),
      inputs_(call_ctx.inputs()),
      outputs_(shared_to_weak(call_ctx.outputs())),
      global_tensor_infer_result_(call_ctx.global_tensor_infer_result()),
      op_interp_ctx_(call_ctx.op_interp_ctx()),
      tmp_tensor_(call_ctx.tmp_tensor()) {
  for (const auto& x : call_ctx.outputs()) {
    ebo_infos_.push_back(EBOInfo{std::make_shared<MemoryCase>(x->mem_case()), x->tensor_meta(),
                                 x->mut_tensor_meta(), x->data_type(), x->memory_format()});
  }
}

CallContext::CallContext(const DtrCallContext& dtr_call_ctx)
    : composed_attrs_(dtr_call_ctx.composed_attrs_),
      inputs_(dtr_call_ctx.inputs_),
      global_tensor_infer_result_(dtr_call_ctx.global_tensor_infer_result_),
      op_interp_ctx_(dtr_call_ctx.op_interp_ctx_),
      tmp_tensor_(dtr_call_ctx.tmp_tensor_) {
  for (int i = 0; i < dtr_call_ctx.outputs_.size(); ++i) {
    const auto& weak = dtr_call_ctx.outputs_[i];
    if (weak.expired()) {
      LOG(INFO) << "index: " << i << " is expired";
      outputs_.push_back(std::make_shared<vm::EagerBlobObject>(
          dtr_call_ctx.ebo_infos_[i].mem_case, dtr_call_ctx.ebo_infos_[i].local_tensor_meta,
          dtr_call_ctx.ebo_infos_[i].dynamic_local_tensor_meta,
          dtr_call_ctx.ebo_infos_[i].data_type, dtr_call_ctx.ebo_infos_[i].memory_format,
          std::make_shared<vm::TensorStorage>(
              true, dtr_call_ctx.ebo_infos_[i].local_tensor_meta->device())));
    } else {
      outputs_.push_back(weak.lock());
    }
  }
}

CallContext DtrCallContext::ToCallContext() const { return CallContext(*this); }

}  // namespace eager
}  // namespace oneflow
