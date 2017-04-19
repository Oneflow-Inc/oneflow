#include "gflags/gflags.h"
#include "job/id_manager.h"
#include "graph/task_graph_manager.h"
#include "job/job_conf.pb.h"

DEFINE_string(job_user_conf_filepath, "", "");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  oneflow::JobUserConf job_user_conf;
  ParseProtoFromTextFile(FLAGS_job_user_conf_filepath, &job_user_conf);
  oneflow::JobDesc::Singleton().Init(job_user_conf);
  oneflow::IDMgr::Singleton().InitFromResource(oneflow::JobDesc::Singleton().resource());
  oneflow::TaskGraphMgr::Singleton().Init();
  return 0;
}
