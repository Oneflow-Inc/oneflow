#include <gflags/gflags.h>
#include <glog/logging.h>
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/ctrl_server.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/persistence/file_system.h"

namespace oneflow {

class Scheduler final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Scheduler);
  ~Scheduler() = default;

  OF_SINGLETON(Scheduler);

  void Process(const std::string& job_conf_filepath,
               const std::string& this_machine_name, char** env);

 private:
  Scheduler() = default;
  void NewAllSingleton(const std::string& job_conf_filepath,
                       const std::string& this_machine_name, char** env);
  void DeleteAllSingleton();
  void SystemCall(const std::string& cmd);

  std::unique_ptr<CtrlServer> ctrl_server_;
  std::string env_prefix_;
};

void Scheduler::Process(const std::string& job_conf_filepath,
                        const std::string& this_machine_name, char** env) {
  NewAllSingleton(job_conf_filepath, this_machine_name, env);
  auto plan = of_make_unique<Plan>();
  std::string naive_plan_filepath = JoinPath(LogDir(), "naive_plan");
  //#ifdef PLATFORM_WINDOWS
  //  char current_dir[128];
  //  GetCurrentDirectory(100, current_dir);
  //  LOG(INFO) << "current_dir is " << current_dir << "\n";
  //#endif  // PLATFORM_WINDOWS
  // Compile
  if (RuntimeCtx::Singleton()->IsThisMachineMaster()) {
    std::stringstream compile_cmd;
#ifdef PLATFORM_WINDOWS
    compile_cmd << "compiler.exe "
                << "-job_conf_filepath=\"" << job_conf_filepath << "\" "
                << "-plan_filepath=\"" << naive_plan_filepath << "\"";
#else
    compile_cmd << "./compiler "
                << "-job_conf_filepath=" << job_conf_filepath << " "
                << "-plan_filepath=" << naive_plan_filepath;
#endif  // PLATFORM_WINDOWS
    SystemCall(compile_cmd.str());
    ParseProtoFromTextFile(naive_plan_filepath, plan.get());
    CtrlClient::Singleton()->PushPlan(*plan);
  } else {
    CtrlClient::Singleton()->PullPlan(plan.get());
  }
  OF_BARRIER();
  if (RuntimeCtx::Singleton()->IsThisMachineMaster()) {
    CtrlClient::Singleton()->ClearPlan();
  } else {
    PrintProtoToTextFile(*plan, naive_plan_filepath);
  }
  // Runtime
  std::stringstream runtime_cmd;
#ifdef PLATFORM_WINDOWS
  runtime_cmd << "runtime.exe "
              << "-plan_filepath=\"" << naive_plan_filepath << "\" "
              << "-this_machine_name=\"" << this_machine_name << "\"";
#else
  runtime_cmd << "./runtime "
              << "-plan_filepath=" << naive_plan_filepath << " "
              << "-this_machine_name=" << this_machine_name;
#endif  // PLATFORM_WINDOWS
  SystemCall(runtime_cmd.str());
  DeleteAllSingleton();
}

void Scheduler::NewAllSingleton(const std::string& job_conf_filepath,
                                const std::string& this_machine_name,
                                char** env) {
  oneflow::JobConf job_conf;
  oneflow::ParseProtoFromTextFile(job_conf_filepath, &job_conf);
  JobDesc::NewSingleton(job_conf);
  IDMgr::NewSingleton();
  RuntimeCtx::NewSingleton(this_machine_name);
  ctrl_server_.reset(
      new CtrlServer(RuntimeCtx::Singleton()->GetThisCtrlAddr()));
  CtrlClient::NewSingleton();
  env_prefix_ = "";
  std::stringstream ss;
  while (*env) {
    LOG(INFO) << *env;
    ss << (*env++) << " ";
  }
  env_prefix_ = ss.str();
}

void Scheduler::DeleteAllSingleton() {
  CtrlClient::DeleteSingleton();
  ctrl_server_.reset();
  RuntimeCtx::DeleteSingleton();
  IDMgr::DeleteSingleton();
  JobDesc::DeleteSingleton();
}

void Scheduler::SystemCall(const std::string& cmd) {
  LOG(INFO) << "SystemCall: [" << cmd << "]";
  CHECK_EQ(std::system(cmd.c_str()), 0);
}

}  // namespace oneflow

DEFINE_string(job_conf_filepath, "", "");
DEFINE_string(this_machine_name, "", "");

int main(int argc, char** argv, char** env) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  oneflow::LocalFS()->CreateDirIfNotExist(oneflow::LogDir());
  oneflow::RedirectStdoutAndStderrToGlogDir();
  LOG(INFO) << "Scheduler Start";
  oneflow::Scheduler::NewSingleton();
  oneflow::Scheduler::Singleton()->Process(FLAGS_job_conf_filepath,
                                           FLAGS_this_machine_name, env);
  oneflow::Scheduler::DeleteSingleton();
  oneflow::CloseStdoutAndStderr();
  LOG(INFO) << "Scheduler Stop";
  return 0;
}
