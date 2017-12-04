#include <gflags/gflags.h>
#include <glog/logging.h>
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/ctrl_server.h"
#include "oneflow/core/job/compiler.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/persistence/file_system.h"

namespace oneflow {

class Oneflow final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Oneflow);
  ~Oneflow() = default;

  OF_SINGLETON(Oneflow);

  void Flow(const JobConf& job_conf, const std::string& this_machine_name);

 private:
  Oneflow() = default;

  std::unique_ptr<CtrlServer> ctrl_server_;
};

void Oneflow::Flow(const JobConf& job_conf,
                   const std::string& this_machine_name) {
  JobDesc::NewSingleton(job_conf);
  IDMgr::NewSingleton();
  int64_t this_machine_id =
      IDMgr::Singleton()->MachineID4MachineName(this_machine_name);
  // Compile
  TodoPlan plan;
  if (this_machine_id == 0) {
    Compiler::NewSingleton();
    plan = Compiler::Singleton()->Compile();
    // CtrlClient::Singleton()->PushPlan(plan);
  } else {
    // CtrlClient::Singleton()->PullPlan(plan.get());
  }
  std::string naive_plan_filepath = JoinPath(LogDir(), "naive_plan");
  PrintProtoToTextFile(plan, naive_plan_filepath);
  IDMgr::DeleteSingleton();
  JobDesc::DeleteSingleton();
}

}  // namespace oneflow

DEFINE_string(job_conf_filepath, "", "");
DEFINE_string(this_machine_name, "", "");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  oneflow::LocalFS()->CreateDirIfNotExist(oneflow::LogDir());
  oneflow::RedirectStdoutAndStderrToGlogDir();
  LOG(INFO) << "Oneflow Start";
  oneflow::JobConf job_conf;
  oneflow::ParseProtoFromTextFile(FLAGS_job_conf_filepath, &job_conf);
  oneflow::Oneflow::NewSingleton();
  oneflow::Oneflow::Singleton()->Flow(job_conf, FLAGS_this_machine_name);
  oneflow::Oneflow::DeleteSingleton();
  LOG(INFO) << "Oneflow Stop";
  oneflow::CloseStdoutAndStderr();
  return 0;
}
