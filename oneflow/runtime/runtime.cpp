#include "gflags/gflags.h"
#include "glog/logging.h"
#include "common/id_manager.h"
#include "common/ofelf.pb.h"
#include "common/protobuf.h"
#include "common/job_desc.h"
#include "runtime/runtime_info.h"

namespace oneflow {

class Runtime final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Runtime);
  ~Runtime() = default;

  static Runtime& Singleton() {
    static Runtime obj;
    return obj;
  }

  void Run(const OfElf& elf, const std::string& this_machine_name) {
    JobDesc::Singleton().InitFromProto(elf.job_desc());
    IDMgr::Singleton().InitFromResource(JobDesc::Singleton().resource());
    RuntimeInfo::Singleton().set_this_machine_name(this_machine_name);
    TODO();
  }

 private:
  Runtime() = default;

};

} // namespace oneflow

DEFINE_string(elf_filepath, "", "");
DEFINE_string(this_machine_name, "", "");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "Runtime Starting Up...";
  oneflow::OfElf elf;
  oneflow::ParseProtoFromTextFile(FLAGS_elf_filepath, &elf);
  oneflow::Runtime::Singleton().Run(elf, FLAGS_this_machine_name);
  LOG(INFO) << "Runtime Shutting Down...";
  return 0;
}
