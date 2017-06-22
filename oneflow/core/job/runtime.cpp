#include "gflags/gflags.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/runtime_info.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/register/register_manager.h"

namespace oneflow {

class Runtime final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Runtime);
  ~Runtime() = default;

  OF_SINGLETON(Runtime);

  void Run(const Plan& plan, const std::string& this_machine_name) {
    JobDesc::Singleton().InitFromProto(plan.job_desc());
    IDMgr::Singleton().InitFromResource(JobDesc::Singleton().resource());
    RuntimeInfo::Singleton().set_this_machine_name(this_machine_name);
    TODO();
  }

 private:
  Runtime() = default;

};

} // namespace oneflow

DEFINE_string(plan_filepath, "", "");
DEFINE_string(this_machine_name, "", "");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "Runtime Starting Up...";
  oneflow::Plan plan;
  oneflow::ParseProtoFromTextFile(FLAGS_plan_filepath, &plan);
  oneflow::Runtime::Singleton().Run(plan, FLAGS_this_machine_name);
  LOG(INFO) << "Runtime Shutting Down...";
  return 0;
}
