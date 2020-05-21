#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job/oneflow.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/core/job/session_global_objects_scope.h"
#include "oneflow/core/job/env.pb.h"
#include "oneflow/core/job/cluster.h"
#include "oneflow/core/control/cluster_control.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/ctrl_server.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"

namespace oneflow {

namespace {

Maybe<void> Run(const std::string& env_proto_filepath) {
  EnvProto env_proto;
  ParseProtoFromTextFile(env_proto_filepath, &env_proto);
  CHECK_ISNULL_OR_RETURN(Global<EnvGlobalObjectsScope>::Get());
  // Global<T>::New is not allowed to be called here
  // because glog is not constructed yet and LOG(INFO) has bad bahavior
  Global<EnvGlobalObjectsScope>::SetAllocated(new EnvGlobalObjectsScope());
  JUST(Global<EnvGlobalObjectsScope>::Get()->Init(env_proto));
  JUST(Cluster::WorkerLoop());
  return Maybe<void>::Ok();
}

}  // namespace

}  // namespace oneflow

DEFINE_string(env_proto, "", "EnvProto file path");

int main(int argc, char* argv[]) {
  using namespace oneflow;
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  CHECK_JUST(Run(FLAGS_env_proto));
  return 0;
}
