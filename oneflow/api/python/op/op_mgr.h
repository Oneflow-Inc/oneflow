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
#ifndef ONEFLOW_API_PYTHON_OP_OP_MGR_H_
#define ONEFLOW_API_PYTHON_OP_OP_MGR_H_

#include "oneflow/api/python/framework/framework.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/operator/op_attribute.pb.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/eager/eager_symbol_storage.h"

namespace oneflow {

inline Maybe<bool> IsOpTypeCaseCpuSupportOnly(int64_t op_type_case) {
  using OnlyCpuSupport = OnlyCpuSupportPredicator;
  CHECK_OR_RETURN((IsClassRegistered<int32_t, OnlyCpuSupport>(op_type_case)))
      << ": op_type_case = " << op_type_case;
  return static_cast<bool>(
      *std::unique_ptr<OnlyCpuSupport>(NewObj<int32_t, OnlyCpuSupport>(op_type_case)));
}

inline Maybe<bool> IsOpTypeNameCpuSupportOnly(const std::string& op_type_name) {
  const user_op::OpRegistryResult* val =
      user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(op_type_name);
  CHECK_OR_RETURN(val != nullptr) << "op_type_name " << op_type_name << " not register";
  return val->cpu_only_supported;
}

inline Maybe<long long> GetUserOpAttrType(const std::string& op_type_name,
                                          const std::string& attr_name) {
  return JUST(GetAttrTypeImpl(op_type_name, attr_name));
}

inline Maybe<std::string> InferOpConf(const std::string& op_conf_str,
                                      const std::string& upstream_signature_str) {
  OperatorConf op_conf;
  CHECK_OR_RETURN(TxtString2PbMessage(op_conf_str, &op_conf)) << "OperatorConf parse failed";
  CHECK_OR_RETURN(op_conf.has_scope_symbol_id());
  OpNodeSignature upstream_signature;
  CHECK_OR_RETURN(TxtString2PbMessage(upstream_signature_str, &upstream_signature))
      << "OpNodeSignature parse failed";
  const auto& scope_storage = *Global<vm::SymbolStorage<Scope>>::Get();
  const auto& scope = scope_storage.Get(op_conf.scope_symbol_id());
  const auto& op = JUST(ConstructAndInferOp(op_conf, upstream_signature, scope));
  const auto& op_attribute = op->GetOpAttributeWithoutOpNameAndLbn();
  return PbMessage2TxtString(*op_attribute);
}

inline Maybe<std::string> GetSerializedOpAttributes() {
  OpAttributeList op_attribute_list;
  const JobSet& job_set = JUST(GetJobSet());
  for (int i = 0; i < job_set.job_size(); i++) {
    const Job& job = job_set.job(i);
    auto scope = std::make_unique<GlobalJobDescScope>(job.job_conf(), i);
    const auto& op_graph = JUST(OpGraph::New(job));
    op_graph->ForEachNode([&op_attribute_list](OpNode* op_node) {
      const auto& op_attribute = op_node->op().GetOpAttributeWithoutOpNameAndLbn();
      op_attribute_list.mutable_op_attribute()->Add()->CopyFrom(*op_attribute);
    });
  }
  return PbMessage2TxtString(op_attribute_list);
}

inline Maybe<bool> IsInterfaceOpTypeCase(int64_t op_type_case) {
  return oneflow::IsClassRegistered<int32_t, oneflow::IsInterfaceOpConf4OpTypeCase>(op_type_case);
}

inline Maybe<long long> GetOpParallelSymbolId(const std::string& op_conf_str) {
  OperatorConf op_conf;
  CHECK_OR_RETURN(TxtString2PbMessage(op_conf_str, &op_conf)) << "OperatorConf parse failed";
  CHECK_OR_RETURN(op_conf.has_scope_symbol_id());
  const auto& scope = Global<vm::SymbolStorage<Scope>>::Get()->Get(op_conf.scope_symbol_id());
  return JUST(scope.GetParallelDescSymbolId(op_conf));
}

inline Maybe<std::string> CheckAndCompleteUserOpConf(const std::string& op_conf_str) {
  OperatorConf op_conf;
  CHECK_OR_RETURN(TxtString2PbMessage(op_conf_str, &op_conf)) << "operator conf parse failed";
  return PbMessage2TxtString(*JUST(CheckAndCompleteUserOpConfImpl(op_conf)));
}

}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_OP_OP_MGR_H_
