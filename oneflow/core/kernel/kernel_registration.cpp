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
#include "oneflow/core/kernel/kernel_registration.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace kernel_registration {

namespace {

HashMap<OperatorConf::OpTypeCase, std::vector<KernelRegistryVal>>* MutKernelRegistry() {
  static HashMap<OperatorConf::OpTypeCase, std::vector<KernelRegistryVal>> creators;
  return &creators;
}

}  // namespace

KernelRegistrarBuilder& KernelRegistrarBuilder::SetCreateFn(CreateFn fn) {
  registry_val_.func = fn;
  return *this;
}

KernelRegistrarBuilder& KernelRegistrarBuilder::SetIsMatchedPred(IsMatchedPredicator fn) {
  registry_val_.cons.SetIsMatchedPred(fn);
  return *this;
}

void KernelRegistrarBuilder::Finalize(OperatorConf::OpTypeCase* op_type,
                                      KernelRegistryVal* val) const {
  *op_type = op_type_;
  val->func = registry_val_.func;
  val->cons = registry_val_.cons;
}

KernelRegistrar::KernelRegistrar(const KernelRegistrarBuilder& builder) {
  auto* creators = MutKernelRegistry();
  OperatorConf::OpTypeCase op_type;
  KernelRegistryVal val;
  builder.Finalize(&op_type, &val);
  (*creators)[op_type].emplace_back(std::move(val));
}

Kernel* CreateKernel(const KernelConf& kernel_conf) {
  auto op_type = kernel_conf.op_attribute().op_conf().op_type_case();
  auto kernel_registry = MutKernelRegistry();
  if (kernel_registry->find(op_type) == kernel_registry->end()) { return nullptr; }
  const auto& registry_vals = kernel_registry->at(op_type);

  Kernel* ret = nullptr;
  bool is_matched = false;
  for (const KernelRegistryVal& val : registry_vals) {
    if (val.cons.IsMatched(kernel_conf)) {
      CHECK(!is_matched)
          << "There are more than one kernel constraints satisfied by kernel conf of "
          << static_cast<size_t>(op_type);
      is_matched = true;
      ret = val.func();
    }
  }
  // TODO: print more info when failed
  return ret;
}

}  // namespace kernel_registration

}  // namespace oneflow
