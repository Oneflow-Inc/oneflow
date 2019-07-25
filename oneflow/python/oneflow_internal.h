#include <google/protobuf/text_format.h>
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/oneflow.h"
#include "oneflow/core/job/foreign_job_instance.h"
#include "oneflow/core/job/environment_objects_scope.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/oneflow.h"
#include "oneflow/core/control/cluster_control.pb.h"
#include "oneflow/core/control/ctrl_client.h"

void InitBySerializedConfigProto(const std::string& config_proto_str) {
  using namespace oneflow;
  ConfigProto config_proto;
  CHECK(google::protobuf::TextFormat::ParseFromString(config_proto_str, &config_proto));
  CHECK_ISNULL(Global<EnvironmentObjectsScope>::Get());
  // Global<T>::New is not allowed to be called here
  // because glog is not constructed yet and LOG(INFO) has bad bahavior
  Global<EnvironmentObjectsScope>::SetAllocated(new EnvironmentObjectsScope(config_proto));
  LOG(INFO) << "NewGlobal " << typeid(EnvironmentObjectsScope).name();
  if (Global<MachineCtx>::Get()->IsThisMachineMaster()) { return; }
  while (true) {
    {
      ClusterControlProto cluster_control;
      Global<CtrlClient>::Get()->PullKV("halt_or_session_start", &cluster_control);
      if (cluster_control.cmd() == kClusterCtrlCmdHalt) { exit(0); }
      CHECK_EQ(cluster_control.cmd(), kClusterCtrlCmdSessionStart);
    }
    JobSet job_set;
    Global<CtrlClient>::Get()->PullKV("session_job_set", &job_set);
    Global<Oneflow>::New(job_set);
    {
      ClusterControlProto cluster_control;
      Global<CtrlClient>::Get()->PullKV("session_end", &cluster_control);
      CHECK_EQ(cluster_control.cmd(), kClusterCtrlCmdSessionEnd);
      Global<Oneflow>::Delete();
    }
  }
}

void InitGlobalOneflowBySerializedJobSet(const std::string& job_set_str) {
  using namespace oneflow;
  CHECK(Global<MachineCtx>::Get()->IsThisMachineMaster());
  ClusterControlProto cluster_control;
  cluster_control.set_cmd(kClusterCtrlCmdSessionStart);
  Global<CtrlClient>::Get()->PushKV("halt_or_session_start", cluster_control);
  JobSet job_set;
  CHECK(google::protobuf::TextFormat::ParseFromString(job_set_str, &job_set));
  CHECK_ISNULL(Global<Oneflow>::Get());
  Global<CtrlClient>::Get()->PushKV("session_job_set", job_set);
  Global<Oneflow>::New(job_set);
}

std::string GetSerializedInterUserJobInfo() {
  using namespace oneflow;
  CHECK(Global<MachineCtx>::Get()->IsThisMachineMaster());
  CHECK_NOTNULL(Global<Oneflow>::Get());
  CHECK_NOTNULL(Global<InterUserJobInfo>::Get());
  std::string ret;
  google::protobuf::TextFormat::PrintToString(*Global<InterUserJobInfo>::Get(), &ret);
  return ret;
}

void LaunchJob(const std::shared_ptr<oneflow::ForeignJobInstance>& cb) {
  using namespace oneflow;
  CHECK(Global<MachineCtx>::Get()->IsThisMachineMaster());
  CHECK_NOTNULL(Global<Oneflow>::Get());
  const auto& job_name = cb->job_name();
  auto* buffer_mgr = Global<BufferMgr<std::shared_ptr<ForeignJobInstance>>>::Get();
  int64_t job_id = Global<JobName2JobId>::Get()->at(job_name);
  const JobDesc& job_desc = GlobalJobDesc(job_id);
  if (job_desc.is_pull_job()) { buffer_mgr->Get(GetForeignOutputBufferName(job_name))->Send(cb); }
  if (job_desc.is_push_job()) { buffer_mgr->Get(GetForeignInputBufferName(job_name))->Send(cb); }
  buffer_mgr->Get(GetCallbackNotifierBufferName(job_name))->Send(cb);
  Global<BufferMgr<int64_t>>::Get()->Get(kBufferNameGlobalWaitJobId)->Send(job_id);
}

void DestroyGlobalOneflow() {
  using namespace oneflow;
  CHECK(Global<MachineCtx>::Get()->IsThisMachineMaster());
  CHECK_NOTNULL(Global<Oneflow>::Get());
  ClusterControlProto cluster_control;
  cluster_control.set_cmd(kClusterCtrlCmdSessionEnd);
  Global<CtrlClient>::Get()->PushKV("session_end", cluster_control);
  Global<Oneflow>::Delete();
}

int Ofblob_GetDataType(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->data_type();
}

size_t OfBlob_NumAxes(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->NumAxes();
}

void OfBlob_CopyShapeToNumpy(uint64_t of_blob_ptr, int64_t* array, int size) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->CopyShapeTo(array, size);
};

namespace oneflow {

namespace {

struct GlobalChecker final {
  GlobalChecker() = default;
  ~GlobalChecker() {
    if (Global<Oneflow>::Get() != nullptr) { LOG(FATAL) << "global oneflow is not destroyed yet"; }
    if (Global<EnvironmentObjectsScope>::Get() != nullptr) {
      if (Global<MachineCtx>::Get()->IsThisMachineMaster()) {
	ClusterControlProto cluster_control;
	cluster_control.set_cmd(kClusterCtrlCmdHalt);
	Global<CtrlClient>::Get()->PushKV("session_end", cluster_control);
      }
      Global<EnvironmentObjectsScope>::Delete();
    }
  }
};

GlobalChecker checker;

}  // namespace

}
