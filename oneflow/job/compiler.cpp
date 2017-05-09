#include "gflags/gflags.h"
#include "glog/logging.h"
#include "job/id_manager.h"
#include "graph/task_graph_manager.h"
#include "job/job_conf.pb.h"
#include "job/ofelf.pb.h"

DEFINE_string(job_conf_filepath, "", "");
DEFINE_string(elf_filepath, "", "");

namespace oneflow {

class Compiler final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Compiler);
  ~Compiler() = default;

  static Compiler& Singleton() {
    static Compiler obj;
    return obj;
  }

  void Compile(const JobConf& job_conf) {
    JobDesc::Singleton().InitFromJobConf(job_conf);
    IDMgr::Singleton().InitFromResource(JobDesc::Singleton().resource());
    TaskGraphMgr::Singleton().BuildGraphs();
    JobDesc::Singleton().set_piece_size(50); // TODO: set appropriate piece_size
    TaskGraphMgr::Singleton().InferShape4Regsts();
    // To Proto
    OfElf elf;
    TaskGraphMgr::Singleton().AllTaskNodesToProto(elf.mutable_tasks());
    RegstDescMgr::Singleton().AllRegstsToProto(elf.mutable_regst_descs());
    OpMgr::Singleton().AllOpToProto(elf.mutable_operators());
    JobDesc::Singleton().ToProto(elf.mutable_job_desc());
    PrintProtoToTextFile(elf, FLAGS_elf_filepath);
  }

 private:
  Compiler() = default;

};

} // namespace oneflow

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "Compiler Starting Up...";
  oneflow::JobConf job_conf;
  oneflow::ParseProtoFromTextFile(FLAGS_job_conf_filepath, &job_conf);
  oneflow::Compiler::Singleton().Compile(job_conf);
  LOG(INFO) << "Compiler Shutting Down...";
  return 0;
}
