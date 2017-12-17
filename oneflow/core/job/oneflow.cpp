#include "oneflow/core/common/str_util.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/ctrl_server.h"
#include "oneflow/core/job/compiler.h"
#include "oneflow/core/job/improver.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/runtime.h"
#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/actor/act_event_logger.h"

namespace oneflow {

class Oneflow final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Oneflow);
  ~Oneflow() = default;

  OF_SINGLETON(Oneflow);

 private:
  Oneflow(const JobConf& job_conf, const std::string& this_mchn_name);

  std::unique_ptr<CtrlServer> ctrl_server_;
};

Oneflow::Oneflow(const JobConf& job_conf, const std::string& this_mchn_name) {
  // New All Singleton
  JobDesc::NewSingleton(job_conf);
  IDMgr::NewSingleton();
  MachineCtx::NewSingleton(this_mchn_name);
  const MachineCtx* machine_ctx = MachineCtx::Singleton();
  ctrl_server_.reset(new CtrlServer(machine_ctx->GetThisCtrlAddr()));
  CtrlClient::NewSingleton();
  // Compile
  Plan plan;
  if (machine_ctx->IsThisMachineMaster()) {
    Compiler::NewSingleton();
    plan = Compiler::Singleton()->Compile();
    CtrlClient::Singleton()->PushKV("naive_plan", plan);
    Compiler::DeleteSingleton();
  } else {
    CtrlClient::Singleton()->PullKV("naive_plan", &plan);
  }
  OF_BARRIER();
  PrintProtoToTextFile(plan, JoinPath(LogDir(), "naive_plan"));
  // Experiment Runtime
  Runtime::NewSingleton(plan, true);
  Runtime::DeleteSingleton();
  // Improve
  if (machine_ctx->IsThisMachineMaster()) {
    Improver::NewSingleton();
    plan = Improver::Singleton()->Improve(
        plan, JoinPath(LogDir(), ActEventLogger::act_event_bin_filename_));
    Improver::DeleteSingleton();
    CtrlClient::Singleton()->PushKV("improved_plan", plan);
  } else {
    CtrlClient::Singleton()->PullKV("improved_plan", &plan);
  }
  OF_BARRIER();
  PrintProtoToTextFile(plan, JoinPath(LogDir(), "improved_plan"));
  CtrlClient::Singleton()->Clear();
  // Runtime
  Runtime::NewSingleton(plan, false);
  Runtime::DeleteSingleton();
  // Delete All Singleton
  CtrlClient::DeleteSingleton();
  ctrl_server_.reset();
  MachineCtx::DeleteSingleton();
  IDMgr::DeleteSingleton();
  JobDesc::DeleteSingleton();
}

}  // namespace oneflow

DEFINE_string(job_conf_filepath, "", "");
DEFINE_string(this_machine_name, "", "");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  oneflow::LocalFS()->CreateDirIfNotExist(oneflow::LogDir());
  oneflow::RedirectStdoutAndStderrToGlogDir();
  oneflow::JobConf job_conf;
  oneflow::ParseProtoFromTextFile(FLAGS_job_conf_filepath, &job_conf);
  oneflow::Oneflow::NewSingleton(job_conf, FLAGS_this_machine_name);
  oneflow::Oneflow::DeleteSingleton();
  oneflow::CloseStdoutAndStderr();
  return 0;
}
