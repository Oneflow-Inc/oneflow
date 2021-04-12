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
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/user/kernels/stateful_opkernel.h"

namespace oneflow {
namespace one {

UserOpExpr::UserOpExpr(const std::string& op_name, UserOpConf&& proto,
                       const std::vector<std::string>& indexed_ibns,
                       const std::vector<std::string>& indexed_obns)
    : BuiltinOpExpr("user", op_name, indexed_ibns, indexed_obns), proto_(std::move(proto)) {
  OperatorConf op_conf;
  BuildOpConf(&op_conf);
  // TODO: set current device tag
  op_conf.set_device_tag("gpu");
  auto mem_case = MemoryCaseUtil::MakeMemCase(DeviceType::kGPU, 0);
  kernel_ = std::make_shared<StatefulOpKernel>(
      std::shared_ptr<const JobDesc>(&GlobalJobDesc(), [](const JobDesc*) {}), op_conf, mem_case,
      &indexed_input_pairs(), &indexed_output_pairs());
}
}  // namespace one
}  // namespace oneflow
