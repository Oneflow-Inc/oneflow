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
#include <iostream>
#include <google/protobuf/text_format.h>
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/env.pb.h"
#include "oneflow/core/job/oneflow.h"
#include "oneflow/core/job/foreign_job_instance.h"
#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/core/job/session_global_objects_scope.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/oneflow.h"
#include "oneflow/core/job/runtime_job_descs.h"
#include "oneflow/core/control/cluster_control.pb.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/cluster_control.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/foreign_watcher.h"
#include "oneflow/core/job/foreign_callback.h"
#include "oneflow/core/job/cluster.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/framework/config_def.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/vm/instruction.pb.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/id_util.h"
#include "oneflow/core/eager/eager_util.h"
#include "oneflow/core/eager/eager_symbol_storage.h"

#ifdef WITH_TENSORRT
#include "oneflow/xrt/api.h"
#endif  // WITH_TENSORRT

namespace oneflow {

Maybe<void> RegisterForeignCallbackOnlyOnce(ForeignCallback* callback) {
  CHECK_ISNULL_OR_RETURN(Global<ForeignCallback>::Get()) << "foreign callback registered";
  Global<ForeignCallback>::SetAllocated(callback);
  return Maybe<void>::Ok();
}

Maybe<void> RegisterWatcherOnlyOnce(ForeignWatcher* watcher) {
  CHECK_ISNULL_OR_RETURN(Global<ForeignWatcher>::Get()) << "foreign watcher registered";
  Global<ForeignWatcher>::SetAllocated(watcher);
  return Maybe<void>::Ok();
}

Maybe<bool> IsOpTypeCaseCpuSupportOnly(int64_t op_type_case) {
  using OnlyCpuSupport = OnlyCpuSupportPredicator;
  CHECK_OR_RETURN(IsClassRegistered<OnlyCpuSupport>(op_type_case))
      << ": op_type_case = " << op_type_case;
  return static_cast<bool>(*std::unique_ptr<OnlyCpuSupport>(NewObj<OnlyCpuSupport>(op_type_case)));
}

Maybe<bool> IsOpTypeNameCpuSupportOnly(const std::string& op_type_name) {
  const user_op::OpRegistryResult* val =
      user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(op_type_name);
  CHECK_OR_RETURN(val != nullptr) << "op_type_name " << op_type_name << " not register";
  return val->cpu_only_supported;
}

Maybe<std::string> CurrentResource() {
  CHECK_NOTNULL_OR_RETURN((Global<ResourceDesc, ForSession>::Get()));
  return PbMessage2TxtString(Global<ResourceDesc, ForSession>::Get()->resource());
}

Maybe<std::string> EnvResource() {
  CHECK_NOTNULL_OR_RETURN((Global<ResourceDesc, ForEnv>::Get()));
  return PbMessage2TxtString(Global<ResourceDesc, ForEnv>::Get()->resource());
}

Maybe<void> InitEnv(const std::string& env_proto_str) {
  EnvProto env_proto;
  CHECK_OR_RETURN(TxtString2PbMessage(env_proto_str, &env_proto))
      << "failed to parse env_proto" << env_proto_str;
  CHECK_ISNULL_OR_RETURN(Global<EnvGlobalObjectsScope>::Get());
  // Global<T>::New is not allowed to be called here
  // because glog is not constructed yet and LOG(INFO) has bad bahavior
  Global<EnvGlobalObjectsScope>::SetAllocated(new EnvGlobalObjectsScope());
  JUST(Global<EnvGlobalObjectsScope>::Get()->Init(env_proto));
  if (!Global<MachineCtx>::Get()->IsThisMachineMaster()) { CHECK_JUST(Cluster::WorkerLoop()); }
  return Maybe<void>::Ok();
}

Maybe<void> DestroyEnv() {
  if (Global<EnvGlobalObjectsScope>::Get() == nullptr) { return Maybe<void>::Ok(); }
  CHECK_OR_RETURN(Global<MachineCtx>::Get()->IsThisMachineMaster());
  ClusterControl::MasterSendHalt();
  return Maybe<void>::Ok();
}

void FixCpuDeviceNum(ConfigProto* config_proto) {
  if (config_proto->resource().cpu_device_num() > 0) { return; }
  config_proto->mutable_resource()->set_cpu_device_num(std::thread::hardware_concurrency());
}

Maybe<void> InitGlobalSession(const std::string& config_proto_str) {
  CHECK_NOTNULL_OR_RETURN(Global<EnvDesc>::Get()) << "env not found";
  CHECK_OR_RETURN(Global<MachineCtx>::Get()->IsThisMachineMaster());

  ClusterControl::MasterSendSessionStart();

  ConfigProto config_proto;
  CHECK_OR_RETURN(TxtString2PbMessage(config_proto_str, &config_proto))
      << "failed to parse config_proto: " << config_proto_str;
  FixCpuDeviceNum(&config_proto);
  Global<CtrlClient>::Get()->PushKV("config_proto", config_proto);

  CHECK_ISNULL_OR_RETURN(Global<SessionGlobalObjectsScope>::Get());
  Global<SessionGlobalObjectsScope>::SetAllocated(new SessionGlobalObjectsScope());
  JUST(Global<SessionGlobalObjectsScope>::Get()->Init(config_proto));
  LOG(INFO) << "NewGlobal " << typeid(SessionGlobalObjectsScope).name();
  return Maybe<void>::Ok();
}

Maybe<void> DestroyGlobalSession() {
  if (Global<SessionGlobalObjectsScope>::Get() == nullptr) { return Maybe<void>::Ok(); }
  CHECK_OR_RETURN(Global<MachineCtx>::Get()->IsThisMachineMaster());
  Global<SessionGlobalObjectsScope>::Delete();
  return Maybe<void>::Ok();
}

Maybe<void> StartGlobalSession() {
  CHECK_NOTNULL_OR_RETURN(Global<SessionGlobalObjectsScope>::Get()) << "session not found";
  CHECK_OR_RETURN(Global<MachineCtx>::Get()->IsThisMachineMaster());
  const JobSet& job_set = Global<LazyJobBuildAndInferCtxMgr>::Get()->job_set();
  if (Global<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
    TeePersistentLogStream::Create("job_set.prototxt")->Write(job_set);
  }
  if (job_set.job().empty()) { return Error::JobSetEmpty() << "no function defined"; }
  CHECK_ISNULL_OR_RETURN(Global<Oneflow>::Get());
  Global<CtrlClient>::Get()->PushKV("session_job_set", job_set);
  Global<const InterJobReuseMemStrategy>::New(job_set.inter_job_reuse_mem_strategy());
  Global<Oneflow>::New();
  JUST(Global<Oneflow>::Get()->Init(job_set));
  return Maybe<void>::Ok();
}

Maybe<std::string> GetSerializedStructureGraph() {
  const auto* job_ctx_mgr = Global<LazyJobBuildAndInferCtxMgr>::Get();
  CHECK_NOTNULL_OR_RETURN(job_ctx_mgr);
  return job_ctx_mgr->structure_graph();
}

Maybe<void> StopGlobalSession() {
  if (Global<Oneflow>::Get() == nullptr) { return Maybe<void>::Ok(); }
  CHECK_OR_RETURN(Global<MachineCtx>::Get()->IsThisMachineMaster());
  CHECK_NOTNULL_OR_RETURN(Global<Oneflow>::Get());
  Global<Oneflow>::Delete();
  Global<const InterJobReuseMemStrategy>::Delete();
  return Maybe<void>::Ok();
}

Maybe<std::string> GetSerializedInterUserJobInfo() {
  CHECK_OR_RETURN(Global<MachineCtx>::Get()->IsThisMachineMaster());
  CHECK_NOTNULL_OR_RETURN(Global<Oneflow>::Get());
  CHECK_NOTNULL_OR_RETURN(Global<InterUserJobInfo>::Get());
  std::string ret;
  google::protobuf::TextFormat::PrintToString(*Global<InterUserJobInfo>::Get(), &ret);
  return ret;
}

Maybe<std::string> GetSerializedJobSet() {
  const auto* job_ctx_mgr = Global<LazyJobBuildAndInferCtxMgr>::Get();
  CHECK_NOTNULL_OR_RETURN(job_ctx_mgr);
  return PbMessage2TxtString(job_ctx_mgr->job_set());
}

Maybe<std::string> GetFunctionConfigDef() {
  std::string ret;
  google::protobuf::TextFormat::PrintToString(GlobalFunctionConfigDef(), &ret);
  return ret;
}

Maybe<void> LaunchJob(const std::shared_ptr<oneflow::ForeignJobInstance>& cb) {
  CHECK_OR_RETURN(Global<MachineCtx>::Get()->IsThisMachineMaster());
  CHECK_NOTNULL_OR_RETURN(Global<Oneflow>::Get());
  const auto& job_name = cb->job_name();
  auto* buffer_mgr = Global<BufferMgr<std::shared_ptr<ForeignJobInstance>>>::Get();
  int64_t job_id = Global<JobName2JobId>::Get()->at(job_name);
  if (IsPullJob(job_name, *Global<InterUserJobInfo>::Get())) {
    buffer_mgr->Get(GetForeignOutputBufferName(job_name))->Send(cb);
  }
  if (IsPushJob(job_name, *Global<InterUserJobInfo>::Get())) {
    buffer_mgr->Get(GetForeignInputBufferName(job_name))->Send(cb);
  }
  buffer_mgr->Get(GetCallbackNotifierBufferName(job_name))->Send(cb);
  Global<BufferMgr<int64_t>>::Get()->Get(kBufferNameGlobalWaitJobId)->Send(job_id);
  return Maybe<void>::Ok();
}

Maybe<std::string> GetSerializedMachineId2DeviceIdListOFRecord(
    const std::string& parallel_conf_str) {
  ParallelConf parallel_conf;
  CHECK_OR_RETURN(TxtString2PbMessage(parallel_conf_str, &parallel_conf))
      << "parallel conf parse failed";
  return PbMessage2TxtString(*JUST(ParseMachineAndDeviceIdList(parallel_conf)));
}

Maybe<void> CacheInt8Calibration() {
#ifdef WITH_TENSORRT
  xrt::tensorrt::CacheInt8Calibration();
#else
  CHECK_OR_RETURN(0) << "Please recompile with TensorRT.";
#endif  // WITH_TENSORRT
  return Maybe<void>::Ok();
}

Maybe<void> WriteInt8Calibration(const std::string& path) {
#ifdef WITH_TENSORRT
  xrt::tensorrt::CacheInt8Calibration();
  xrt::tensorrt::WriteInt8Calibration(path);
#else
  CHECK_OR_RETURN(0) << "Please recompile with TensorRT.";
#endif  // WITH_TENSORRT
  return Maybe<void>::Ok();
}

Maybe<long long> GetUserOpAttrType(const std::string& op_type_name, const std::string& attr_name) {
  return JUST(GetUserOpAttrTypeImpl(op_type_name, attr_name));
}

Maybe<std::string> CheckAndCompleteUserOpConf(const std::string& op_conf_str) {
  OperatorConf op_conf;
  CHECK_OR_RETURN(TxtString2PbMessage(op_conf_str, &op_conf)) << "operator conf parse failed";
  return PbMessage2TxtString(*JUST(CheckAndCompleteUserOpConfImpl(op_conf)));
}

Maybe<std::string> InferOpConf(const std::string& op_conf_str,
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

Maybe<long> GetOpParallelSymbolId(const std::string& op_conf_str) {
  OperatorConf op_conf;
  CHECK_OR_RETURN(TxtString2PbMessage(op_conf_str, &op_conf)) << "OperatorConf parse failed";
  CHECK_OR_RETURN(op_conf.has_scope_symbol_id());
  const auto& scope = Global<vm::SymbolStorage<Scope>>::Get()->Get(op_conf.scope_symbol_id());
  return JUST(scope.GetParallelDescSymbolId(op_conf));
}

Maybe<void> RunLogicalInstruction(const std::string& instruction_list_str,
                                  const std::string& eager_symbol_list_str) {
  return eager::RunLogicalInstruction(instruction_list_str, eager_symbol_list_str);
}

Maybe<void> RunPhysicalInstruction(const std::string& instruction_list_str,
                                   const std::string& eager_symbol_list_str) {
  return eager::RunPhysicalInstruction(instruction_list_str, eager_symbol_list_str);
}

Maybe<long long> CurrentMachineId() {
  CHECK_NOTNULL_OR_RETURN(Global<MachineCtx>::Get());
  return Global<MachineCtx>::Get()->this_machine_id();
}

Maybe<long long> NewLogicalObjectId() {
  CHECK_OR_RETURN(JUST(GlobalMaybe<MachineCtx>())->IsThisMachineMaster());
  return vm::IdUtil::NewLogicalObjectId();
}

Maybe<long long> NewLogicalSymbolId() {
  CHECK_OR_RETURN(JUST(GlobalMaybe<MachineCtx>())->IsThisMachineMaster());
  return vm::IdUtil::NewLogicalSymbolId();
}

Maybe<long long> NewPhysicalObjectId() {
  CHECK_NOTNULL_OR_RETURN(Global<MachineCtx>::Get());
  return vm::IdUtil::NewPhysicalObjectId(Global<MachineCtx>::Get()->this_machine_id());
}

Maybe<long long> NewPhysicalSymbolId() {
  CHECK_NOTNULL_OR_RETURN(Global<MachineCtx>::Get());
  return vm::IdUtil::NewPhysicalSymbolId(Global<MachineCtx>::Get()->this_machine_id());
}

}  // namespace oneflow
