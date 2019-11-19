#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job/oneflow.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/cluster_objects_scope.h"
#include "oneflow/core/job/environment_objects_scope.h"
#include "oneflow/core/job/cluster.pb.h"
#include "oneflow/core/control/cluster_control.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/ctrl_server.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"

namespace oneflow {

namespace {

Maybe<void> Run(const std::string& config_proto_filepath) {
  ClusterProto cluster_proto;
  ParseProtoFromTextFile(config_proto_filepath, &cluster_proto);
  OF_CHECK_ISNULL(Global<ClusterObjectsScope>::Get());
  // Global<T>::New is not allowed to be called here
  // because glog is not constructed yet and LOG(INFO) has bad bahavior
  Global<ClusterObjectsScope>::SetAllocated(new ClusterObjectsScope());
  JUST(Global<ClusterObjectsScope>::Get()->Init(cluster_proto));
  OF_CHECK(!Global<MachineCtx>::Get()->IsThisMachineMaster());
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
  return Maybe<void>::Ok();
}

}  // namespace

}  // namespace oneflow

DEFINE_string(cluster_proto, "", "ClusterProto file path");

int main(int argc, char* argv[]) {
  using namespace oneflow;
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  CHECK_JUST(Run(FLAGS_cluster_proto));
  return 0;
}
