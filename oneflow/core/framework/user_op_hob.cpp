#include <sstream>

#include "oneflow/core/framework/kernel_registration.h"
#include "oneflow/core/framework/user_op_hob.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {

namespace user_op {

bool KernelRegBoolFunctor::operator()(const KernelRegContext& ctx) const { return match_fn_(ctx); }

std::string KernelRegBoolFunctor::DebugStr(const KernelRegContext& ctx, bool display_result) const {
  std::ostringstream string_stream;
  string_stream << "\"" << debug_str_ << "\"";
  if (display_result) {
    std::string boolResult = match_fn_(ctx) ? "True" : "False";
    string_stream << "[" << boolResult << "]";
  }
  return string_stream.str();
}

hob::BoolFunctorPtr<KernelRegContext> HobDeviceTypeEq(DeviceType device_type) {
  std::ostringstream string_stream;
  string_stream << "device_type == " << *CHECK_JUST(DeviceTag4DeviceType(device_type));
  const std::shared_ptr<const hob::BoolFunctor<KernelRegContext>> krbf_ptr =
      std::make_shared<const KernelRegBoolFunctor>(
          string_stream.str(),
          [device_type](const KernelRegContext& ctx) { return ctx.device_type() == device_type; });
  return krbf_ptr;
}

hob::BoolFunctorPtr<KernelRegContext> HobDataTypeEq(const std::string& tensor_name, int tensor_idx,
                                                    DataType data_type) {
  std::ostringstream string_stream;
  string_stream << "tensor \'" << tensor_name << "\' data_type == " << DataType_Name(data_type);
  const std::shared_ptr<const hob::BoolFunctor<KernelRegContext>> krbf_ptr =
      std::make_shared<const KernelRegBoolFunctor>(
          string_stream.str(), [tensor_name, tensor_idx, data_type](const KernelRegContext& ctx) {
            const user_op::TensorDesc* desc =
                ctx.TensorDesc4ArgNameAndIndex(tensor_name, tensor_idx);
            return desc->data_type() == data_type;
          });
  return krbf_ptr;
}

}  // namespace user_op

}  // namespace oneflow
