#include "oneflow/core/eager/call_context.h"

namespace oneflow {
namespace eager {
namespace {
vm::EagerBlobObjectList weak_to_shared(
    const small_vector<std::weak_ptr<vm::EagerBlobObject>, kOpArgsReservedSize>& weak_list) {
  vm::EagerBlobObjectList ret;
  ret.reserve(weak_list.size());
  for (const auto& weak : weak_list) {
    CHECK(!weak.expired());
    ret.emplace_back(CHECK_NOTNULL(weak.lock()));
  }
  return ret;
}

small_vector<std::weak_ptr<vm::EagerBlobObject>, kOpArgsReservedSize> shared_to_weak(
    const vm::EagerBlobObjectList& shared_list) {
  small_vector<std::weak_ptr<vm::EagerBlobObject>, kOpArgsReservedSize> ret;
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
      tmp_tensor_(call_ctx.tmp_tensor()) {}

CallContext::CallContext(const DtrCallContext& dtr_call_ctx)
    : composed_attrs_(dtr_call_ctx.composed_attrs_),
      inputs_(dtr_call_ctx.inputs_),
      outputs_(weak_to_shared(dtr_call_ctx.outputs_)),
      global_tensor_infer_result_(dtr_call_ctx.global_tensor_infer_result_),
      op_interp_ctx_(dtr_call_ctx.op_interp_ctx_),
      tmp_tensor_(dtr_call_ctx.tmp_tensor_) {}

}  // namespace eager
}  // namespace oneflow
