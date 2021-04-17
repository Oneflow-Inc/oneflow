#include "oneflow/core/eager/local_call_opkernel_phy_instr_operand.h"
#include "oneflow/user/kernels/stateful_local_opkernel.h"

namespace oneflow {
namespace eager {
void LocalCallOpKernelPhyInstrOperand::ForEachInferMutMirroredObject(
                const std::function<void(vm::MirroredObject*)>& fn) const {
  for (const int64_t input_index : opkernel_->input_tuple_indexes4mut_ibns()) {
    inputs_->at(input_index)
  }
}
}  // namespace eager
}  // namespace oneflow
