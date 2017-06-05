#include "gflags/gflags.h"
#include "glog/logging.h"
#include "oneflow/core/common/id_manager.h"
#include "oneflow/core/common/ofelf.pb.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/job_desc.h"
#include "oneflow/core/runtime/runtime_info.h"
#include "oneflow/core/register/register_manager.h"

namespace oneflow {

class ElfRunner final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ElfRunner);
  ~ElfRunner() = default;

  static ElfRunner& Singleton() {
    static ElfRunner obj;
    return obj;
  }

  void Run(const OfElf& elf, const std::string& this_machine_name) {
    JobDesc::Singleton().InitFromProto(elf.job_desc());
    IDMgr::Singleton().InitFromResource(JobDesc::Singleton().resource());
    RuntimeInfo::Singleton().set_this_machine_name(this_machine_name);
    TODO();
  }

 private:
  ElfRunner() = default;

};

} // namespace oneflow

DEFINE_string(elf_filepath, "", "");
DEFINE_string(this_machine_name, "", "");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "ElfRunner Starting Up...";
  oneflow::OfElf elf;
  oneflow::ParseProtoFromTextFile(FLAGS_elf_filepath, &elf);
  oneflow::ElfRunner::Singleton().Run(elf, FLAGS_this_machine_name);
  LOG(INFO) << "ElfRunner Shutting Down...";
  return 0;
}
