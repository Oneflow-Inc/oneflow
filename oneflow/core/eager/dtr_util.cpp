#include "oneflow/core/eager/dtr_util.h"

#include <algorithm>
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/eager/local_call_opkernel_phy_instr_operand.h"
#include "oneflow/user/kernels/stateful_local_opkernel.h"

namespace oneflow {
namespace vm {

namespace {

auto ConvertToDTRVector(const std::vector<std::shared_ptr<EagerBlobObject>>& base_class_vector) {
  std::vector<std::shared_ptr<DTREagerBlobObject>> sub_class_vector;
  std::transform(base_class_vector.begin(), base_class_vector.end(),
                 std::back_inserter(sub_class_vector),
                 [](const std::shared_ptr<EagerBlobObject>& x) {
                   return CHECK_NOTNULL(std::dynamic_pointer_cast<DTREagerBlobObject>(x));
                 });
  return sub_class_vector;
};

}  // namespace

std::vector<std::shared_ptr<DTREagerBlobObject>> GetDTRInputs(
    const std::shared_ptr<const LocalCallOpKernelPhyInstrOperand>& operand) {
  return GetDTRInputs(operand.get());
}

std::vector<std::shared_ptr<DTREagerBlobObject>> GetDTROutputs(
    const std::shared_ptr<const LocalCallOpKernelPhyInstrOperand>& operand) {
  return GetDTROutputs(operand.get());
}

std::vector<std::shared_ptr<DTREagerBlobObject>> GetDTRInputs(
    const LocalCallOpKernelPhyInstrOperand* operand) {
  return ConvertToDTRVector(*operand->inputs());
}

std::vector<std::shared_ptr<DTREagerBlobObject>> GetDTROutputs(
    const LocalCallOpKernelPhyInstrOperand* operand) {
  return ConvertToDTRVector(*operand->outputs());
}

std::shared_ptr<LocalCallOpKernelPhyInstrOperand> DTROp2LocalCallOp(DTRInstrOperand* operand) {
  const auto& inputs = operand->inputs();
  const auto& outputs = operand->outputs();

  std::shared_ptr<one::EagerBlobObjectList> input_shared_ptr =
      std::make_shared<one::EagerBlobObjectList>(inputs.size());
  std::shared_ptr<one::EagerBlobObjectList> output_shared_ptr =
      std::make_shared<one::EagerBlobObjectList>(outputs.size());

  for (int i = 0; i < inputs.size(); ++i) {
    if (auto input = inputs[i].lock()) {
      input_shared_ptr->at(i) = input;
    } else {
      // CHECK_JUST(Global<one::DTRTensorPool>::Get()->display2());
      LOG(FATAL) << "null at input " << i << " of op "
                 << operand->shared_opkernel()->op_type_name();
    }
  }

  for (int i = 0; i < outputs.size(); ++i) {
    if (auto output = outputs[i].lock()) {
      output_shared_ptr->at(i) = output;
    } else {
      // CHECK_JUST(Global<one::DTRTensorPool>::Get()->display2());
      LOG(FATAL) << "null at output " << i << " of op "
                 << operand->shared_opkernel()->op_type_name();
    }
  }

  auto phy_instr_operand = std::make_shared<LocalCallOpKernelPhyInstrOperand>(
      operand->shared_opkernel(), input_shared_ptr, output_shared_ptr,
      operand->consistent_tensor_infer_result(), operand->op_interp_ctx(),
      operand->dev_vm_dep_object_consume_mode());

  return phy_instr_operand;
}

namespace {
Maybe<void> CheckInMemory(const std::vector<std::shared_ptr<DTREagerBlobObject>>& vec) {
  int i = 0;
  for (auto& dtr_blob_object : vec) {
    if (dtr_blob_object->blob().shape().elem_cnt() > 0) {
      CHECK_OR_RETURN(dtr_blob_object->is_in_memory());
      CHECK_NOTNULL_OR_RETURN(dtr_blob_object->blob().dptr());
    }
    i++;
  }
  return Maybe<void>::Ok();
}
}

Maybe<void> CheckInputInMemory(LocalCallOpKernelPhyInstrOperand* operand) {
  return CheckInMemory(GetDTRInputs(operand));
}

Maybe<void> CheckOutputInMemory(LocalCallOpKernelPhyInstrOperand* operand) {
  return CheckInMemory(GetDTROutputs(operand));
}

}  // namespace vm
}  // namespace oneflow
