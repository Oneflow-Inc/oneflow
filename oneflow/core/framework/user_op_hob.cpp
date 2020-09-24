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

hob::HobContextGetter<KernelRegContext, DataType> HobDataType(const std::string& tensor_name,
                                                              int tensor_idx) {
  std::ostringstream string_stream;
  string_stream << "data_type of tensor \'" << tensor_name << "\'";
  return hob::HobContextGetter<KernelRegContext, DataType>(
      string_stream.str(), [tensor_name, tensor_idx](const KernelRegContext& ctx) {
        const user_op::TensorDesc* desc = ctx.TensorDesc4ArgNameAndIndex(tensor_name, tensor_idx);
        return desc->data_type();
      });
}

HobStringContextGetter<KernelRegContext> HobDeviceTag() {
  std::ostringstream string_stream;
  string_stream << "device_tag";
  return HobStringContextGetter<KernelRegContext>(
      string_stream.str(),
      [](const KernelRegContext& ctx) -> const std::string& { return ctx.device_tag(); });
}

}  // namespace user_op

}  // namespace oneflow
