#include "gflags/gflags.h"
#include "glog/logging.h"
#include "job/id_manager.h"
#include "graph/task_graph_manager.h"
#include "job/job_conf.pb.h"

using oneflow::JobConf;
using oneflow::JobDesc;
using oneflow::IDMgr;
using oneflow::TaskGraphMgr;

DEFINE_string(job_conf_filepath, "", "");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "Compiler Starting Up...";
  LOG(INFO) << "Read JobConf...";
  JobConf job_conf;
  ParseProtoFromTextFile(FLAGS_job_conf_filepath, &job_conf);
  JobDesc::Singleton().InitFromJobConf(job_conf);
  IDMgr::Singleton().InitFromResource(JobDesc::Singleton().resource());
  TaskGraphMgr::Singleton().Init();
  LOG(INFO) << "Compiler Shutting Down...";
  return 0;
}
