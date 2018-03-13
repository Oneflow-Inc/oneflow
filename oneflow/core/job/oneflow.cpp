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
  AvailableMemDescOfMachine this_machine_mem_desc;
#ifdef WITH_CUDA
  const JobDesc* job_desc = JobDesc::Singleton();
  FOR_RANGE(int, i, 0, job_desc->GpuDeviceNum()) {
    this_machine_mem_desc.add_zone_size(GetAvailableGpuMemSize(i));
  }
#endif
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

void FixCpuDeviceNum() {
  int32_t cpu_device_num = JobDesc::Singleton()->CpuDeviceNum();
  if (cpu_device_num > 0) { return; }
  if (MachineCtx::Singleton()->IsThisMachineMaster()) {
    cpu_device_num = std::thread::hardware_concurrency();
    CtrlClient::Singleton()->PushKVT("cpu_device_num", cpu_device_num);
  } else {
    CtrlClient::Singleton()->PullKVT("cpu_device_num", &cpu_device_num);
  }
  OF_BARRIER();
  CtrlClient::Singleton()->ClearKV("cpu_device_num");
  CHECK_GT(cpu_device_num, 0);
  JobDesc::Singleton()->SetCpuDeviceNum(cpu_device_num);
}

}  // namespace

class Oneflow final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Oneflow);
  ~Oneflow() = default;

  OF_SINGLETON(Oneflow);

 private:
  Oneflow(const JobDescProto& job_desc, const std::string& this_mchn_name);

  std::unique_ptr<CtrlServer> ctrl_server_;
};

Oneflow::Oneflow(const JobDescProto& job_desc,
                 const std::string& this_mchn_name) {
  // New All Singleton
  JobDesc::NewSingleton(job_desc);
  IDMgr::NewSingleton();
  MachineCtx::NewSingleton(this_mchn_name);
  const MachineCtx* machine_ctx = MachineCtx::Singleton();
  ctrl_server_.reset(new CtrlServer(machine_ctx->GetThisCtrlAddr()));
  CtrlClient::NewSingleton();
  FixCpuDeviceNum();
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
DEFINE_string(job_desc_filepath, "", "");
DEFINE_string(this_machine_name, "", "");

int main(int argc, char** argv) {
  using namespace oneflow;
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LocalFS()->CreateDirIfNotExist(LogDir());
  RedirectStdoutAndStderrToGlogDir();
  JobDescProto job_desc;
  if (FLAGS_job_desc_filepath != "") {
    ParseProtoFromTextFile(FLAGS_job_desc_filepath, &job_desc);
  } else if (FLAGS_job_conf_filepath != "") {
    JobConf* jc = job_desc.mutable_job_conf();
    ParseProtoFromTextFile(FLAGS_job_conf_filepath, jc);
    ParseProtoFromTextFile(jc->dlnet_filepath(), job_desc.mutable_dlnet_conf());
    ParseProtoFromTextFile(jc->resource_filepath(),
                           job_desc.mutable_resource());
    ParseProtoFromTextFile(jc->placement_filepath(),
                           job_desc.mutable_placement());
  } else {
    LOG(FATAL) << "Please Set job_conf_filepath or job_desc_filepath";
  }
  Oneflow::NewSingleton(job_desc, FLAGS_this_machine_name);
  Oneflow::DeleteSingleton();
  CloseStdoutAndStderr();
  return 0;
}
