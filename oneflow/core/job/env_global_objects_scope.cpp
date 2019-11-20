#include "oneflow/core/job/env_global_objects_scope.h"
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

Maybe<void> EnvGlobalObjectsScope::Init(const EnvProto& env_proto) {
  InitLogging(env_proto.cpp_logging_conf());
  Global<EnvDesc>::New(env_proto);
  Global<CtrlServer>::New();
  Global<CtrlClient>::New();
  int64_t this_mchn_id =
      Global<EnvDesc>::Get()->GetMachineId(Global<CtrlServer>::Get()->this_machine_addr());
  Global<MachineCtx>::New(this_mchn_id);
  return Maybe<void>::Ok();
}

EnvGlobalObjectsScope::~EnvGlobalObjectsScope() {
  CHECK_NOTNULL(Global<MachineCtx>::Get());
  CHECK_NOTNULL(Global<CtrlClient>::Get());
  CHECK_NOTNULL(Global<CtrlServer>::Get());
  CHECK_NOTNULL(Global<EnvDesc>::Get());
  Global<MachineCtx>::Delete();
  Global<CtrlClient>::Delete();
  Global<CtrlServer>::Delete();
  Global<EnvDesc>::Delete();
}

}  // namespace oneflow
