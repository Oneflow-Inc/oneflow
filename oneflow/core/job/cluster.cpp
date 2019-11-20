#include "oneflow/core/job/cluster.h"
#include "oneflow/core/control/cluster_control.pb.h"
#include "oneflow/core/control/cluster_control.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/oneflow.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/session_global_objects_scope.h"
#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/core/job/job_set.pb.h"

namespace oneflow {

Maybe<void> Cluster::WorkerLoop() {
  OF_CHECK(!Global<MachineCtx>::Get()->IsThisMachineMaster());
  while (ClusterControl::WorkerReceiveHalt() == false) {
    ConfigProto config_proto;
    Global<CtrlClient>::Get()->PullKV("config_proto", &config_proto);
    int32_t machine_num = config_proto.resource().machine_num();
    if (Global<MachineCtx>::Get()->this_machine_id() >= machine_num) { continue; }
    Global<SessionGlobalObjectsScope>::New();
    JUST(Global<SessionGlobalObjectsScope>::Get()->Init(config_proto));

    JobSet job_set;
    Global<CtrlClient>::Get()->PullKV("session_job_set", &job_set);
    { Oneflow oneflow(job_set); }
    Global<SessionGlobalObjectsScope>::Delete();
  }
  ClusterControl::HaltBarrier();
  Global<EnvGlobalObjectsScope>::Delete();
  exit(0);
}

}  // namespace oneflow
