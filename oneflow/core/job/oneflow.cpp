#include "oneflow/core/common/str_util.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/ctrl_server.h"
#include "oneflow/core/job/compiler.h"
#include "oneflow/core/job/improver.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/runtime.h"
#include "oneflow/core/job/available_memory_desc.pb.h"
#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/actor/act_event_logger.h"

namespace oneflow {

namespace {

std::string GetAmdCtrlKey(int64_t machine_id) {
  return "AvailableMemDesc/" + std::to_string(machine_id);
}

void PushAvailableMemDescOfThisMachine() {
  const JobDesc* job_desc = JobDesc::Singleton();
  AvailableMemDescOfMachine this_machine_mem_desc;
  if (job_desc->GetDeviceType() == DeviceType::kGPU) {
    FOR_RANGE(int, i, 0, job_desc->resource().device_num_per_machine()) {
      this_machine_mem_desc.add_zone_size(GetAvailableGpuMemSize(i));
    }
  }
  this_machine_mem_desc.add_zone_size(GetAvailableCpuMemSize());
  CtrlClient::Singleton()->PushKV(
      GetAmdCtrlKey(MachineCtx::Singleton()->this_machine_id()),
      this_machine_mem_desc);
}

AvailableMemDesc PullAvailableMemDesc() {
  AvailableMemDesc ret;
  AvailableMemDescOfMachine machine_amd_i;
  FOR_RANGE(int64_t, i, 0, JobDesc::Singleton()->TotalMachineNum()) {
    CtrlClient::Singleton()->PullKV(GetAmdCtrlKey(i), ret.add_machine_amd());
  }
  return ret;
}

}  // namespace

class Oneflow final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Oneflow);
  ~Oneflow() = default;

  OF_SINGLETON(Oneflow);

 private:
  Oneflow(const JobConf& job_conf, const std::string& this_mchn_name);

  std::unique_ptr<CtrlServer> ctrl_server_;
};

Oneflow::Oneflow(const JobConf& job_conf, const std::string& this_mchn_name) {
  // New All Singleton
  JobDesc::NewSingleton(job_conf);
  IDMgr::NewSingleton();
  MachineCtx::NewSingleton(this_mchn_name);
  const MachineCtx* machine_ctx = MachineCtx::Singleton();
  ctrl_server_.reset(new CtrlServer(machine_ctx->GetThisCtrlAddr()));
  CtrlClient::NewSingleton();
  // Compile
  Plan plan;
  if (machine_ctx->IsThisMachineMaster()) {
    Compiler::NewSingleton();
    plan = Compiler::Singleton()->Compile();
    CtrlClient::Singleton()->PushKV("naive_plan", plan);
    Compiler::DeleteSingleton();
  } else {
    CtrlClient::Singleton()->PullKV("naive_plan", &plan);
  }
  OF_BARRIER();
  PrintProtoToTextFile(plan, JoinPath(LogDir(), "naive_plan"));
  // Experiment Runtime
  Runtime::NewSingleton(plan, true);
  Runtime::DeleteSingleton();
  PushAvailableMemDescOfThisMachine();
  // Improve
  if (machine_ctx->IsThisMachineMaster()) {
    Improver::NewSingleton(PullAvailableMemDesc());
    plan = Improver::Singleton()->Improve(
        plan, JoinPath(LogDir(), ActEventLogger::act_event_bin_filename_));
    Improver::DeleteSingleton();
    CtrlClient::Singleton()->PushKV("improved_plan", plan);
  } else {
    CtrlClient::Singleton()->PullKV("improved_plan", &plan);
  }
  OF_BARRIER();
  PrintProtoToTextFile(plan, JoinPath(LogDir(), "improved_plan"));
  CtrlClient::Singleton()->Clear();
  OF_BARRIER();
  // Runtime
  Runtime::NewSingleton(plan, false);
  Runtime::DeleteSingleton();
  // Delete All Singleton
  CtrlClient::DeleteSingleton();
  ctrl_server_.reset();
  MachineCtx::DeleteSingleton();
  IDMgr::DeleteSingleton();
  JobDesc::DeleteSingleton();
}

}  // namespace oneflow

DEFINE_string(job_conf_filepath, "", "");
DEFINE_string(this_machine_name, "", "");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  oneflow::LocalFS()->CreateDirIfNotExist(oneflow::LogDir());
  oneflow::RedirectStdoutAndStderrToGlogDir();
  oneflow::JobConf job_conf;
  oneflow::ParseProtoFromTextFile(FLAGS_job_conf_filepath, &job_conf);
  oneflow::Oneflow::NewSingleton(job_conf, FLAGS_this_machine_name);
  oneflow::Oneflow::DeleteSingleton();
  oneflow::CloseStdoutAndStderr();
  return 0;
}
