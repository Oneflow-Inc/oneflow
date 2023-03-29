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
#include "oneflow/core/framework/user_op_registry_manager.h"

#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/common/tensor_desc.h"
#include "oneflow/core/kernel/kernel.pb.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/common/env_var/env_var.h"

namespace oneflow {

DEFINE_ENV_BOOL(ONEFLOW_KERNEL_ENABLE_PRIORITY_EXPERIMENTAL, false);

namespace user_op {

UserOpRegistryMgr& UserOpRegistryMgr::Get() {
  static UserOpRegistryMgr mgr;
  return mgr;
}

OpRegistry UserOpRegistryMgr::CheckAndGetOpRegistry(const std::string& op_type_name) {
  CHECK(!op_type_name.empty());
  auto it = op_reg_result_.find(op_type_name);
  CHECK(it == op_reg_result_.end());
  return OpRegistry().Name(op_type_name);
}

Maybe<void> UserOpRegistryMgr::Register(OpRegistryResult result) {
  CHECK_OR_RETURN(result.data_type_infer_fn);
  CHECK_OR_RETURN(op_reg_result_.emplace(result.op_type_name, result).second);
  return Maybe<void>::Ok();
}

const OpRegistryResult* UserOpRegistryMgr::GetOpRegistryResult(const std::string& op_type_name) {
  auto it = op_reg_result_.find(op_type_name);
  if (it != op_reg_result_.end()) { return &(it->second); }
  return nullptr;
}

OpKernelRegistry UserOpRegistryMgr::CheckAndGetOpKernelRegistry(const std::string& op_type_name) {
  CHECK(!op_type_name.empty());
  return OpKernelRegistry().Name(op_type_name);
}

Maybe<void> UserOpRegistryMgr::Register(OpKernelRegistryResult result) {
  op_kernel_reg_result_[result.op_type_name].emplace_back(result);
  return Maybe<void>::Ok();
}

namespace {

std::string GetErrorMsgOfSearchedOp(const KernelRegContext& ctx) {
  const auto& op_conf = ctx.user_op_conf();
  std::stringstream ss;
  ss << " The Info of OperatorConf are "
     << "\n op_name: " << op_conf.op_name() << "\n op_type_name: " << op_conf.op_type_name()
     << "\n DeviceType_Name: " << DeviceType_Name(ctx.device_type());
  for (const auto& pair : ctx.inputs()) {
    ss << "\n DataType_Name of " << pair.first << "_" << pair.second << ": "
       << DataType_Name(ctx.TensorDesc4ArgNameAndIndex(pair.first, pair.second)->data_type());
  }
  for (const auto& pair : ctx.outputs()) {
    ss << "\n DataType_Name of " << pair.first << "_" << pair.second << ": "
       << DataType_Name(ctx.TensorDesc4ArgNameAndIndex(pair.first, pair.second)->data_type());
  }
  return ss.str();
}

}  // namespace

Maybe<const OpKernelRegistryResult*> UserOpRegistryMgr::GetOpKernelRegistryResult(
    const std::string& op_type_name, const KernelRegContext& ctx) {
  auto it = op_kernel_reg_result_.find(op_type_name);
  if (it == op_kernel_reg_result_.end()) {
    return Error::OpKernelNotFoundError({})
           << "There is no kernel registered for Current OperatorConf. "
           << GetErrorMsgOfSearchedOp(ctx);
  }

  const OpKernelRegistryResult* ret = nullptr;
  int32_t cur_priority = kKernelPriorityFallback;
  const bool enable_priority_experimental = EnvBool<ONEFLOW_KERNEL_ENABLE_PRIORITY_EXPERIMENTAL>();
  for (const auto& reg_val : it->second) {
    if (reg_val.priority >= kKernelPriorityExperimental && (!enable_priority_experimental)) {
      continue;
    }
    if (reg_val.is_matched_hob->get(ctx)) {
      if (ret == nullptr || reg_val.priority > cur_priority) {
        ret = &reg_val;
        cur_priority = reg_val.priority;
      } else if (ret != nullptr && reg_val.priority == cur_priority) {
        LOG(WARNING)
            << "There are more than one kernels with same priority matching Current OperatorConf. "
            << GetErrorMsgOfSearchedOp(ctx);
      } else {
        // do nothing
      }
    }
  }

  if (ret == nullptr) {
    std::vector<std::string> debug_msgs;
    for (const auto& reg_val : it->second) {
      debug_msgs.emplace_back(reg_val.is_matched_hob->DebugStr(ctx));
    }
    return Error::OpKernelNotFoundError(debug_msgs)
           << "Cannot find the kernel matching Current OperatorConf. "
           << GetErrorMsgOfSearchedOp(ctx);
  }

  return ret;
}

Maybe<bool> UserOpRegistryMgr::IsOpKernelRegistered(const std::string& op_type_name,
                                                    const KernelRegContext& ctx) {
  auto it = op_kernel_reg_result_.find(op_type_name);
  if (it == op_kernel_reg_result_.end()) { return false; }
  const bool enable_priority_experimental = EnvBool<ONEFLOW_KERNEL_ENABLE_PRIORITY_EXPERIMENTAL>();
  for (const auto& reg_val : it->second) {
    if (reg_val.priority >= kKernelPriorityExperimental && (!enable_priority_experimental)) {
      continue;
    }
    if (reg_val.is_matched_hob->get(ctx)) { return true; }
  }
  return false;
}

UserOpHostMemoryInputRegistry& UserOpHostMemoryInputRegistry::Get() {
  static UserOpHostMemoryInputRegistry mgr;
  return mgr;
}

Maybe<void> UserOpHostMemoryInputRegistry::SetHostMemoryInput4Op(const std::string& op_type_name,
                                                                 const std::string& arg_name,
                                                                 int32_t index) {
  auto it = op_type_name2host_memory_input_args_.find(op_type_name);
  if (it == op_type_name2host_memory_input_args_.end()) {
    auto pair = op_type_name2host_memory_input_args_.emplace(
        op_type_name, small_vector<std::pair<std::string, int32_t>>());
    CHECK_OR_RETURN(pair.second);
    it = pair.first;
  }
  it->second.emplace_back(std::make_pair(arg_name, index));
  return Maybe<void>::Ok();
}

bool UserOpHostMemoryInputRegistry::IsHostMemoryInput4Op(const std::string& op_type_name,
                                                         const std::string& arg_name,
                                                         int32_t index) const {
  auto it = op_type_name2host_memory_input_args_.find(op_type_name);
  if (it == op_type_name2host_memory_input_args_.end()) { return false; }
  return std::find(it->second.begin(), it->second.end(), std::make_pair(arg_name, index))
         != it->second.end();
}

bool UserOpHostMemoryInputRegistry::HasHostMemoryInput(const std::string& op_type_name) const {
  return op_type_name2host_memory_input_args_.find(op_type_name)
         != op_type_name2host_memory_input_args_.end();
}

}  // namespace user_op
}  // namespace oneflow
