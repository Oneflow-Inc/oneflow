#include "oneflow/core/job/oneflow.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/environment_objects_scope.h"
#include "oneflow/core/control/cluster_control.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/ctrl_server.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"

namespace oneflow {

namespace {

void Run(const std::string& config_proto_filepath) {
  ConfigProto config_proto;
  ParseProtoFromTextFile(config_proto_filepath, &config_proto);
  // Global<T>::New is not allowed to be called here
  // because glog is not constructed yet and LOG(INFO) has bad bahavior
  Global<EnvironmentObjectsScope>::SetAllocated(new EnvironmentObjectsScope(config_proto));
  LOG(INFO) << "NewGlobal " << typeid(EnvironmentObjectsScope).name();
  CHECK_EQ(Global<MachineCtx>::Get()->IsThisMachineMaster(), false);
  while (ClusterControl::WorkerReceiveHalt() == false) {
    JobSet job_set;
    Global<CtrlClient>::Get()->PullKV("session_job_set", &job_set);
    TeePersistentLogStream::Create("session_job_set")->Write(job_set);
    { Oneflow oneflow(job_set); }
  }
  ClusterControl::WorkerSendHaltAck();
  Global<EnvironmentObjectsScope>::Delete();
}

}  // namespace

}  // namespace oneflow

DEFINE_string(config_proto, "", "ConfigProto file path");

int main(int argc, char* argv[]) {
  using namespace oneflow;
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  Run(FLAGS_config_proto);
  return 0;
}
