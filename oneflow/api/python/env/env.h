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
#ifndef ONEFLOW_API_PYTHON_ENV_ENV_H_
#define ONEFLOW_API_PYTHON_ENV_ENV_H_

#include <string>
#include <google/protobuf/text_format.h>
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/job/cluster_instruction.h"
#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/graph_scope_vars.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/rpc/include/base.h"
#include "oneflow/core/ep/include/device_manager_registry.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/virtual_machine.h"

namespace oneflow {

inline Maybe<std::string> CurrentResource() {
  CHECK_NOTNULL_OR_RETURN((Singleton<ResourceDesc, ForSession>::Get()));
  return PbMessage2TxtString(Singleton<ResourceDesc, ForSession>::Get()->resource());
}

inline Maybe<std::string> EnvResource() {
  CHECK_NOTNULL_OR_RETURN((Singleton<ResourceDesc, ForEnv>::Get()));
  return PbMessage2TxtString(Singleton<ResourceDesc, ForEnv>::Get()->resource());
}

inline Maybe<long long> CurrentMachineId() { return GlobalProcessCtx::Rank(); }

inline Maybe<int64_t> GetRank() { return GlobalProcessCtx::Rank(); }
inline Maybe<size_t> GetWorldSize() { return GlobalProcessCtx::WorldSize(); }
inline Maybe<size_t> GetNodeSize() { return GlobalProcessCtx::NodeSize(); }
inline Maybe<size_t> GetLocalRank() { return GlobalProcessCtx::LocalRank(); }
inline Maybe<size_t> CudaGetDeviceCount() {
  return Singleton<ep::DeviceManagerRegistry>::Get()->GetDeviceCount(DeviceType::kCUDA);
}

inline Maybe<void> SetFLAGS_alsologtostderr(bool flag) {
  FLAGS_alsologtostderr = flag;
  return Maybe<void>::Ok();
}
inline Maybe<bool> GetFLAGS_alsologtostderr() {
  return FLAGS_alsologtostderr;
}  // namespace oneflow
inline Maybe<void> SetFLAGS_v(int32_t v_level) {
  FLAGS_v = v_level;
  return Maybe<void>::Ok();
}
inline Maybe<int32_t> GetFLAGS_v() { return FLAGS_v; }

inline Maybe<void> EmptyCache() {
  JUST(vm::CurrentRankSync());
  auto* vm = JUST(SingletonMaybe<VirtualMachine>());
  JUST(vm->ShrinkAllMem());
  return Maybe<void>::Ok();
}

inline Maybe<void> SetGraphLRVerbose(bool verbose) {
  SetGraphVerboseStepLr(verbose);
  return Maybe<void>::Ok();
}
inline bool GetGraphLRVerbose() { return IsOpenGraphVerboseStepLr(); }

}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_ENV_ENV_H_
