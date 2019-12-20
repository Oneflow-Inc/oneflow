#include <iostream>
#include <google/protobuf/text_format.h>
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/job/job_set.pb.h"
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
#include "oneflow/core/job/cluster.h"
#include "oneflow/core/framework/config_def.h"

namespace oneflow {

Maybe<void> RegisterWatcherOnlyOnce(ForeignWatcher* watcher) {
  OF_CHECK_ISNULL(Global<ForeignWatcher>::Get()) << "foreign watcher registered";
  // no delete
  Global<ForeignWatcher>::SetAllocated(watcher);
  return Maybe<void>::Ok();
}

Maybe<bool> IsOpTypeCaseCpuSupportOnly(int64_t op_type_case) {
  using OnlyCpuSupport = OnlyCpuSupportPredicator;
  OF_CHECK(IsClassRegistered<OnlyCpuSupport>(op_type_case)) << ": op_type_case = " << op_type_case;
  return static_cast<bool>(*std::unique_ptr<OnlyCpuSupport>(NewObj<OnlyCpuSupport>(op_type_case)));
}

Maybe<void> InitEnv(const std::string& env_proto_str) {
  EnvProto env_proto;
  OF_CHECK(TxtString2PbMessage(env_proto_str, &env_proto))
      << "failed to parse env_proto" << env_proto_str;
  OF_CHECK_ISNULL(Global<EnvGlobalObjectsScope>::Get());
  // Global<T>::New is not allowed to be called here
  // because glog is not constructed yet and LOG(INFO) has bad bahavior
  Global<EnvGlobalObjectsScope>::SetAllocated(new EnvGlobalObjectsScope());
  JUST(Global<EnvGlobalObjectsScope>::Get()->Init(env_proto));
  if (!Global<MachineCtx>::Get()->IsThisMachineMaster()) { CHECK_JUST(Cluster::WorkerLoop()); }
  return Maybe<void>::Ok();
}

Maybe<void> DestroyEnv() {
  if (Global<EnvGlobalObjectsScope>::Get() == nullptr) { return Maybe<void>::Ok(); }
  OF_CHECK(Global<MachineCtx>::Get()->IsThisMachineMaster());
  ClusterControl::MasterSendHalt();
  return Maybe<void>::Ok();
}

void FixCpuDeviceNum(ConfigProto* config_proto) {
  if (config_proto->resource().cpu_device_num() > 0) { return; }
  config_proto->mutable_resource()->set_cpu_device_num(std::thread::hardware_concurrency());
}

Maybe<void> InitGlobalSession(const std::string& config_proto_str) {
  OF_CHECK_NOTNULL(Global<EnvDesc>::Get()) << "env not found";
  OF_CHECK(Global<MachineCtx>::Get()->IsThisMachineMaster());

  ClusterControl::MasterSendSessionStart();

  ConfigProto config_proto;
  OF_CHECK(TxtString2PbMessage(config_proto_str, &config_proto))
      << "failed to parse config_proto: " << config_proto_str;
  FixCpuDeviceNum(&config_proto);
  Global<CtrlClient>::Get()->PushKV("config_proto", config_proto);

  OF_CHECK_ISNULL(Global<SessionGlobalObjectsScope>::Get());
  Global<SessionGlobalObjectsScope>::SetAllocated(new SessionGlobalObjectsScope());
  JUST(Global<SessionGlobalObjectsScope>::Get()->Init(config_proto));
  LOG(INFO) << "NewGlobal " << typeid(SessionGlobalObjectsScope).name();
  return Maybe<void>::Ok();
}

Maybe<void> DestroyGlobalSession() {
  if (Global<SessionGlobalObjectsScope>::Get() == nullptr) { return Maybe<void>::Ok(); }
  OF_CHECK(Global<MachineCtx>::Get()->IsThisMachineMaster());
  Global<SessionGlobalObjectsScope>::Delete();
  return Maybe<void>::Ok();
}

Maybe<void> StartGlobalSession() {
  OF_CHECK_NOTNULL(Global<SessionGlobalObjectsScope>::Get()) << "session not found";
  OF_CHECK(Global<MachineCtx>::Get()->IsThisMachineMaster());
  const JobSet& job_set = Global<JobBuildAndInferCtxMgr>::Get()->job_set();
  if (job_set.job().empty()) { return Error::JobSetEmpty() << "no function defined"; }
  OF_CHECK_ISNULL(Global<Oneflow>::Get());
  Global<CtrlClient>::Get()->PushKV("session_job_set", job_set);
  Global<const InterJobReuseMemStrategy>::New(job_set.inter_job_reuse_mem_strategy());
  Global<Oneflow>::New(job_set);
  return Maybe<void>::Ok();
}

Maybe<void> StopGlobalSession() {
  if (Global<Oneflow>::Get() == nullptr) { return Maybe<void>::Ok(); }
  OF_CHECK(Global<MachineCtx>::Get()->IsThisMachineMaster());
  OF_CHECK_NOTNULL(Global<Oneflow>::Get());
  Global<Oneflow>::Delete();
  Global<const InterJobReuseMemStrategy>::Delete();
  return Maybe<void>::Ok();
}

Maybe<std::string> GetSerializedInterUserJobInfo() {
  OF_CHECK(Global<MachineCtx>::Get()->IsThisMachineMaster());
  OF_CHECK_NOTNULL(Global<Oneflow>::Get());
  OF_CHECK_NOTNULL(Global<InterUserJobInfo>::Get());
  std::string ret;
  google::protobuf::TextFormat::PrintToString(*Global<InterUserJobInfo>::Get(), &ret);
  return ret;
}

Maybe<std::string> GetFunctionConfigDef() {
  std::string ret;
  google::protobuf::TextFormat::PrintToString(GlobalFunctionConfigDef(), &ret);
  return ret;
}

Maybe<void> LaunchJob(const std::shared_ptr<oneflow::ForeignJobInstance>& cb) {
  OF_CHECK(Global<MachineCtx>::Get()->IsThisMachineMaster());
  OF_CHECK_NOTNULL(Global<Oneflow>::Get());
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

Maybe<long long> GetDeviceType4DeviceTag(const std::string& device_tag) {
  return *JUST(DeviceType4DeviceTag(device_tag));
}

Maybe<std::string> GetSerializedMachineId2DeviceIdListOFRecord(
    const std::string& parallel_conf_str) {
  ParallelConf parallel_conf;
  OF_CHECK(TxtString2PbMessage(parallel_conf_str, &parallel_conf)) << "parallel conf parse failed";
  return PbMessage2TxtString(*JUST(ParseMachineAndDeviceIdList(parallel_conf)));
}

namespace {

struct GlobalChecker final {
  GlobalChecker() = default;
  ~GlobalChecker() {
    if (Global<Oneflow>::Get() != nullptr) {
      std::cerr << "global session is not closed yet" << std::endl;
    }
    if (Global<SessionGlobalObjectsScope>::Get() != nullptr) {
      std::cerr << "global session is not destroyed yet" << std::endl;
    }
  }
};

GlobalChecker checker;

}  // namespace

}  // namespace oneflow
