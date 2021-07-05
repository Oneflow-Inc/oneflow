#include "oneflow/core/eager/run_job_phy_instr_operand.h"

namespace oneflow {
namespace vm {

void RunJobPhyInstrOperand::ForEachConstMirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
    const {
  for (const auto& input : inputs()) {
    DoEach(nullptr, CHECK_JUST(input->compute_local_dep_object())
                        ->mut_local_dep_object()
                        ->mut_mirrored_object());
  }
}

void RunJobPhyInstrOperand::ForEachMutMirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
    const {
  for (const auto& parameter : parameters()) {
    DoEach(nullptr, CHECK_JUST(parameter->compute_local_dep_object())
                        ->mut_local_dep_object()
                        ->mut_mirrored_object());
  }
}

void RunJobPhyInstrOperand::ForEachMut2MirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
    const {
  // TODO(lixinqi): move partial of outputs into ForEachMutMirroredObject if shape infered before compute.
  for (const auto& output : outputs()) {
    DoEach(nullptr, CHECK_JUST(output->compute_local_dep_object())
                        ->mut_local_dep_object()
                        ->mut_mirrored_object());
  }
}

}
}
