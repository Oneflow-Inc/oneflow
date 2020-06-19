#include <sstream>

#include "oneflow/core/framework/user_op_hob.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {

namespace user_op {

hob::BoolFunctorPtr<KernelRegContext> HobTrue() {
  std::ostringstream string_stream;
  string_stream << "\" always true \"";
  const std::shared_ptr<const hob::BoolFunctor<KernelRegContext>> krbf_ptr =
      std::make_shared<const hob::HighOrderBoolFunctor<KernelRegContext>>(
          string_stream.str(), [](const KernelRegContext& ctx) { return true; });
  return krbf_ptr;
}

hob::BoolFunctorPtr<KernelRegContext> HobFalse() {
  std::ostringstream string_stream;
  string_stream << "\" always false \"";
  const std::shared_ptr<const hob::BoolFunctor<KernelRegContext>> krbf_ptr =
      std::make_shared<const hob::HighOrderBoolFunctor<KernelRegContext>>(
          string_stream.str(), [](const KernelRegContext& ctx) { return false; });
  return krbf_ptr;
}

hob::HobContextGetter<KernelRegContext, DeviceType> HobDeviceType() {
  std::ostringstream string_stream;
  string_stream << "device_type";
  return hob::HobContextGetter<KernelRegContext, DeviceType>(
      string_stream.str(), [](const KernelRegContext& ctx) { return ctx.device_type(); });
}

hob::HobContextGetter<KernelRegContext, DataType> HobDataType(const std::string& tensor_name,
                                                              int tensor_idx) {
  std::ostringstream string_stream;
  string_stream << "tensor \'" << tensor_name << "\' data_type";
  return hob::HobContextGetter<KernelRegContext, DataType>(
      string_stream.str(), [tensor_name, tensor_idx](const KernelRegContext& ctx) {
        const user_op::TensorDesc* desc = ctx.TensorDesc4ArgNameAndIndex(tensor_name, tensor_idx);
        return desc->data_type();
      });
}

}  // namespace user_op

}  // namespace oneflow
