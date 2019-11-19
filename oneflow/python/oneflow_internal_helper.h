#include <google/protobuf/text_format.h>
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/cluster.pb.h"
#include "oneflow/core/job/oneflow.h"
#include "oneflow/core/job/foreign_job_instance.h"
#include "oneflow/core/job/cluster_objects_scope.h"
#include "oneflow/core/job/environment_objects_scope.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/oneflow.h"
#include "oneflow/core/job/runtime_job_descs.h"
#include "oneflow/core/control/cluster_control.pb.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/cluster_control.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/foreign_watcher.h"
#include "oneflow/core/job/session.h"

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

Maybe<void> InitCluster(const std::string& cluster_proto_str) {
  ClusterProto cluster_proto;
  OF_CHECK(TxtString2PbMessage(cluster_proto_str, &cluster_proto))
      << "failed to parse cluster_proto" << cluster_proto_str;
  OF_CHECK_ISNULL(Global<ClusterObjectsScope>::Get());
  // Global<T>::New is not allowed to be called here
  // because glog is not constructed yet and LOG(INFO) has bad bahavior
  Global<ClusterObjectsScope>::SetAllocated(new ClusterObjectsScope());
  JUST(Global<ClusterObjectsScope>::Get()->Init(cluster_proto));
  if (Global<MachineCtx>::Get()->IsThisMachineMaster()) { return Maybe<void>::Ok(); }
  // workers go here as daemons
  while (ClusterControl::WorkerReceiveHalt() == false) {
    ConfigProto config_proto;
    Global<CtrlClient>::Get()->PullKV("config_proto", &config_proto);
    Global<EnvironmentObjectsScope>::SetAllocated(new EnvironmentObjectsScope());
    JUST(Global<EnvironmentObjectsScope>::Get()->Init(config_proto));
    LOG(INFO) << "NewGlobal " << typeid(EnvironmentObjectsScope).name();

    JobSet job_set;
    Global<CtrlClient>::Get()->PullKV("session_job_set", &job_set);
    { Oneflow oneflow(job_set); }
  }
  ClusterControl::WorkerSendHaltAck();
  Global<ClusterObjectsScope>::Delete();
  exit(0);
  // avoid compiler complain
  return Maybe<void>::Ok();
}

Maybe<void> DestroyCluster() {
  if (Global<ClusterObjectsScope>::Get() == nullptr) { return Maybe<void>::Ok(); }
  OF_CHECK(Global<MachineCtx>::Get()->IsThisMachineMaster());
  ClusterControl::MasterSendHaltAndWaitAck();
  return Maybe<void>::Ok();
}

void FixCpuDeviceNum(ConfigProto* config_proto) {
  if (config_proto->resource().cpu_device_num() > 0) { return; }
  config_proto->mutable_resource()->set_cpu_device_num(std::thread::hardware_concurrency());
}

Maybe<void> InitGlobalEnvironment(const std::string& config_proto_str) {
  OF_CHECK_NOTNULL(Global<ClusterDesc>::Get()) << "cluster not found";
  OF_CHECK(Global<MachineCtx>::Get()->IsThisMachineMaster());

  ClusterControl::MasterSendSessionStart();

  ConfigProto config_proto;
  OF_CHECK(TxtString2PbMessage(config_proto_str, &config_proto))
      << "failed to parse config_proto: " << config_proto_str;
  FixCpuDeviceNum(&config_proto);
  Global<CtrlClient>::Get()->PushKV("config_proto", config_proto);

  OF_CHECK_ISNULL(Global<EnvironmentObjectsScope>::Get());
  Global<EnvironmentObjectsScope>::SetAllocated(new EnvironmentObjectsScope());
  JUST(Global<EnvironmentObjectsScope>::Get()->Init(config_proto));
  LOG(INFO) << "NewGlobal " << typeid(EnvironmentObjectsScope).name();
  return Maybe<void>::Ok();
}

Maybe<void> DestroyGlobalEnvironment() {
  if (Global<EnvironmentObjectsScope>::Get() == nullptr) { return Maybe<void>::Ok(); }
  OF_CHECK(Global<MachineCtx>::Get()->IsThisMachineMaster());
  Global<EnvironmentObjectsScope>::Delete();
  return Maybe<void>::Ok();
}

Maybe<void> InitGlobalSession() {
  OF_CHECK_NOTNULL(Global<EnvironmentObjectsScope>::Get()) << "environment not inited";
  OF_CHECK_ISNULL(Global<Session>::Get()) << "no multi sessions supported";
  Global<Session>::New();
  return Maybe<void>::Ok();
}

Maybe<void> DestroyGlobalSession() {
  OF_CHECK_NOTNULL(Global<Session>::Get()) << "session not found";
  Global<Session>::Delete();
  return Maybe<void>::Ok();
}

Maybe<void> InitGlobalOneflow() {
  OF_CHECK_NOTNULL(Global<Session>::Get()) << "session not found";
  OF_CHECK(Global<MachineCtx>::Get()->IsThisMachineMaster());
  const JobSet& job_set = Global<JobBuildAndInferCtxMgr>::Get()->job_set();
  if (job_set.job().empty()) { return Error::JobSetEmpty() << "no function defined"; }
  OF_CHECK_ISNULL(Global<Oneflow>::Get());
  Global<CtrlClient>::Get()->PushKV("session_job_set", job_set);
  Global<const InterJobReuseMemStrategy>::New(job_set.inter_job_reuse_mem_strategy());
  Global<Oneflow>::New(job_set);
  return Maybe<void>::Ok();
}

Maybe<void> DestroyGlobalOneflow() {
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
    if (Global<Oneflow>::Get() != nullptr) { LOG(FATAL) << "global oneflow is not destroyed yet"; }
    if (Global<EnvironmentObjectsScope>::Get() != nullptr) {
      LOG(FATAL) << "global environment is not destroyed yet";
    }
  }
};

GlobalChecker checker;

}  // namespace

}  // namespace oneflow
