#include "gflags/gflags.h"
#include "glog/logging.h"
#include "job/id_manager.h"
#include "job/ofelf.pb.h"
#include "common/protobuf.h"

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
