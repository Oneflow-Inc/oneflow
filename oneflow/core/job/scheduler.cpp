#include <gflags/gflags.h>
#include <glog/logging.h>
#include "oneflow/core/comm_network/ctrl_comm_network.h"
#include "oneflow/core/common/str_util.h"
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
  uint16_t GetNextPort();
  void NewAllSingleton(const std::string& job_conf_filepath,
                       const std::string& this_machine_name, char** env);
  std::string GetEnvPrefix();
  void SystemCall(const std::string& cmd);

  uint16_t next_port_;
  std::string env_prefix_;
};

void Scheduler::Process(const std::string& job_conf_filepath,
                        const std::string& this_machine_name, char** env) {
  NewAllSingleton(job_conf_filepath, this_machine_name, env);
  auto plan = of_make_unique<Plan>();
  std::string naive_plan_filepath = JoinPath(LogDir(), "naive_plan");
  // Compile
  if (RuntimeCtx::Singleton()->IsThisMachineMaster()) {
    std::stringstream compile_cmd;
    compile_cmd << "./compiler "
                << "-job_conf_filepath=" << job_conf_filepath << " "
                << "-plan_filepath=" << naive_plan_filepath;
    SystemCall(compile_cmd.str());
    ParseProtoFromTextFile(naive_plan_filepath, plan.get());
    CtrlCommNet::Singleton()->PublishPlan(plan.get());
  } else {
    CtrlCommNet::Singleton()->FetchPlan(plan.get());
  }
  OF_BARRIER();
  if (RuntimeCtx::Singleton()->IsThisMachineMaster()) {
    CtrlCommNet::Singleton()->PublishPlan(nullptr);
  } else {
    PrintProtoToTextFile(*plan, naive_plan_filepath);
  }
  // Runtime
  std::stringstream runtime_cmd;
  runtime_cmd << "./runtime "
              << "-plan_filepath=" << naive_plan_filepath << " "
              << "-this_machine_name=" << this_machine_name << " "
              << "-ctrl_port=" << GetNextPort() << " "
              << "-data_port=" << GetNextPort();
  SystemCall(runtime_cmd.str());
}

uint16_t Scheduler::GetNextPort() {
  CHECK_LE(next_port_, JobDesc::Singleton()->resource().port_max());
  return next_port_++;
}

void Scheduler::NewAllSingleton(const std::string& job_conf_filepath,
                                const std::string& this_machine_name,
                                char** env) {
  oneflow::JobConf job_conf;
  oneflow::ParseProtoFromTextFile(job_conf_filepath, &job_conf);
  JobDesc::NewSingleton(job_conf);
  next_port_ = JobDesc::Singleton()->resource().port_min();
  IDMgr::NewSingleton();
  RuntimeCtx::NewSingleton(this_machine_name);
  CtrlCommNet::NewSingleton(GetNextPort());
  env_prefix_ = "";
  std::stringstream ss;
  while (*env) {
    LOG(INFO) << *env;
    ss << (*env++) << " ";
  }
  env_prefix_ = ss.str();
}

void Scheduler::SystemCall(const std::string& cmd) {
  LOG(INFO) << "SystemCall: [" << cmd << "]";
  PCHECK(std::system(cmd.c_str()) == 0);
}

}  // namespace oneflow

DEFINE_string(job_conf_filepath, "", "");
DEFINE_string(this_machine_name, "", "");

int main(int argc, char** argv, char** env) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  oneflow::LocalFS()->CreateDirIfNotExist(oneflow::LogDir());
  LOG(INFO) << "Scheduler Start";
  oneflow::Scheduler::NewSingleton();
  oneflow::Scheduler::Singleton()->Process(FLAGS_job_conf_filepath,
                                           FLAGS_this_machine_name, env);
  oneflow::Scheduler::DeleteSingleton();
  LOG(INFO) << "Scheduler Stop";
  return 0;
}
