#include <gflags/gflags.h>
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

class Compiler final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Compiler);
  ~Compiler() = default;

  OF_SINGLETON(Compiler);

  Plan Compile(const JobConf& job_conf);

 private:
  Compiler() = default;

  Plan DoCompile();
};

Plan Compiler::Compile(const JobConf& job_conf) {
  JobDesc::NewSingleton(job_conf);
  IDMgr::NewSingleton();
  OpMgr::NewSingleton();
  LogicalGraph::NewSingleton();
  Plan plan = DoCompile();
  LogicalGraph::DeleteSingleton();
  OpMgr::DeleteSingleton();
  IDMgr::DeleteSingleton();
  JobDesc::DeleteSingleton();
  return plan;
}

Plan Compiler::DoCompile() {
  auto chain_gph = of_make_unique<ChainGraph>(JobDesc::Singleton()->is_train());
  auto task_gph = of_make_unique<TaskGraph>(std::move(chain_gph));
  Plan ret;
  return ret;
}

}  // namespace oneflow

DEFINE_string(job_conf_filepath, "", "");
DEFINE_string(plan_filepath, "", "");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  oneflow::RedirectStdoutAndStderrToGlogDir();
  LOG(INFO) << "Compile Start";
  oneflow::JobConf job_conf;
  oneflow::ParseProtoFromTextFile(FLAGS_job_conf_filepath, &job_conf);
  oneflow::Compiler::NewSingleton();
  oneflow::Plan plan;
  plan = oneflow::Compiler::Singleton()->Compile(job_conf);
  oneflow::PrintProtoToTextFile(plan, FLAGS_plan_filepath);
  oneflow::Compiler::DeleteSingleton();
  oneflow::CloseStdoutAndStderr();
  LOG(INFO) << "Compile Stop";
  return 0;
}
