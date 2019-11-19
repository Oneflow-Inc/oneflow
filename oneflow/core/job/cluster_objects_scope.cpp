#include "oneflow/core/job/cluster_objects_scope.h"
#include "oneflow/core/control/ctrl_server.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/common/str_util.h"

namespace oneflow {

namespace {

std::string LogDir(const std::string& log_dir) {
  char hostname[255];
  CHECK_EQ(gethostname(hostname, sizeof(hostname)), 0);
  std::string v = log_dir + "/" + std::string(hostname);
  return v;
}

void InitLogging(const CppLoggingConf& logging_conf) {
  FLAGS_log_dir = LogDir(logging_conf.log_dir());
  FLAGS_logtostderr = logging_conf.logtostderr();
  FLAGS_logbuflevel = logging_conf.logbuflevel();
  google::InitGoogleLogging("oneflow");
  LocalFS()->RecursivelyCreateDirIfNotExist(FLAGS_log_dir);
}

}  // namespace

Maybe<void> ClusterObjectsScope::Init(const ClusterProto& cluster_proto) {
  InitLogging(cluster_proto.cpp_logging_conf());
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
  CHECK_NOTNULL(Global<MachineCtx>::Get());
  CHECK_NOTNULL(Global<CtrlClient>::Get());
  CHECK_NOTNULL(Global<CtrlServer>::Get());
  CHECK_NOTNULL(Global<ClusterDesc>::Get());
  Global<MachineCtx>::Delete();
  Global<CtrlClient>::Delete();
  Global<CtrlServer>::Delete();
  Global<ClusterDesc>::Delete();
}

}  // namespace oneflow
