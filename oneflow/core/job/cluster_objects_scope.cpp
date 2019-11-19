#include "oneflow/core/job/cluster_objects_scope.h"
#include "oneflow/core/control/ctrl_server.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/machine_context.h"

namespace oneflow {

ClusterObjectsScope::ClusterObjectsScope() {}

Maybe<void> ClusterObjectsScope::Init(const ClusterProto& cluster_proto) {
  Global<ClusterDesc>::New(cluster_proto);
  Global<CtrlServer>::New();
  Global<CtrlClient>::New();
  OF_BARRIER();
  int64_t this_mchn_id =
      Global<ClusterDesc>::Get()->GetMachineId(Global<CtrlServer>::Get()->this_machine_addr());
  Global<MachineCtx>::New(this_mchn_id);
  return Maybe<void>::Ok();
}

ClusterObjectsScope::~ClusterObjectsScope() {
  Global<MachineCtx>::Delete();
  Global<CtrlClient>::Delete();
  Global<CtrlServer>::Delete();
  Global<ClusterDesc>::Delete();
}

}  // namespace oneflow
