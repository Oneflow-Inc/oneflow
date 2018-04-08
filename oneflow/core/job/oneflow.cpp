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
  const JobDesc* job_desc = Global<JobDesc>::Get();
  FOR_RANGE(int, i, 0, job_desc->GpuDeviceNum()) {
    this_machine_mem_desc.add_zone_size(GetAvailableGpuMemSize(i));
  }
#endif
  this_machine_mem_desc.add_zone_size(GetAvailableCpuMemSize());
  Global<CtrlClient>::Get()->PushKV(
      GetAmdCtrlKey(Global<MachineCtx>::Get()->this_machine_id()),
      this_machine_mem_desc);
}

AvailableMemDesc PullAvailableMemDesc() {
  AvailableMemDesc ret;
  AvailableMemDescOfMachine machine_amd_i;
  FOR_RANGE(int64_t, i, 0, Global<JobDesc>::Get()->TotalMachineNum()) {
    Global<CtrlClient>::Get()->PullKV(GetAmdCtrlKey(i), ret.add_machine_amd());
  }
  return ret;
}

void FixCpuDeviceNum() {
  int32_t cpu_device_num = Global<JobDesc>::Get()->CpuDeviceNum();
  if (cpu_device_num > 0) { return; }
  if (Global<MachineCtx>::Get()->IsThisMachineMaster()) {
    cpu_device_num = std::thread::hardware_concurrency();
    Global<CtrlClient>::Get()->PushKVT("cpu_device_num", cpu_device_num);
  } else {
    Global<CtrlClient>::Get()->PullKVT("cpu_device_num", &cpu_device_num);
  }
  OF_BARRIER();
  Global<CtrlClient>::Get()->ClearKV("cpu_device_num");
  CHECK_GT(cpu_device_num, 0);
  Global<JobDesc>::Get()->SetCpuDeviceNum(cpu_device_num);
}

}  // namespace

class Oneflow final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Oneflow);
  ~Oneflow() = default;

  Oneflow(const JobDescProto& job_desc, const std::string& this_mchn_name);

 private:
  std::unique_ptr<CtrlServer> ctrl_server_;
};

Oneflow::Oneflow(const JobDescProto& job_desc,
                 const std::string& this_mchn_name) {
  // New All Global
  Global<JobDesc>::New(job_desc);
  Global<IDMgr>::New();
  Global<MachineCtx>::New(this_mchn_name);
  const MachineCtx* machine_ctx = Global<MachineCtx>::Get();
  ctrl_server_.reset(new CtrlServer(machine_ctx->GetThisCtrlAddr()));
  Global<CtrlClient>::New();
  FixCpuDeviceNum();
  // Compile
  Plan plan;
  if (machine_ctx->IsThisMachineMaster()) {
    Compiler compiler;
    plan = compiler.Compile();
    Global<CtrlClient>::Get()->PushKV("naive_plan", plan);
  } else {
    Global<CtrlClient>::Get()->PullKV("naive_plan", &plan);
  }
  OF_BARRIER();
  PrintProtoToTextFile(plan, JoinPath(LogDir(), "naive_plan"));
  // Experiment Runtime
  { Runtime experiment_run(plan, true); }
  PushAvailableMemDescOfThisMachine();
  // Improve
  if (machine_ctx->IsThisMachineMaster()) {
    const AvailableMemDesc& amd = PullAvailableMemDesc();
    PrintProtoToTextFile(amd, JoinPath(LogDir(), "available_mem_desc"));
    Improver improver(amd);
    plan = improver.Improve(
        plan, JoinPath(LogDir(), ActEventLogger::act_event_bin_filename_));
    Global<CtrlClient>::Get()->PushKV("improved_plan", plan);
  } else {
    Global<CtrlClient>::Get()->PullKV("improved_plan", &plan);
  }
  OF_BARRIER();
  PrintProtoToTextFile(plan, JoinPath(LogDir(), "improved_plan"));
  Global<CtrlClient>::Get()->Clear();
  OF_BARRIER();
  // Runtime
  { Runtime run(plan, false); }
  // Delete All Global
  Global<CtrlClient>::Delete();
  ctrl_server_.reset();
  Global<MachineCtx>::Delete();
  Global<IDMgr>::Delete();
  Global<JobDesc>::Delete();
}

}  // namespace oneflow

DEFINE_string(job_conf_filepath, "", "");
DEFINE_string(job_desc_filepath, "", "");
DEFINE_string(this_machine_name, "", "");

int main(int argc, char** argv) {
  using namespace oneflow;
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LocalFS()->RecursivelyCreateDirIfNotExist(LogDir());
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
  { Oneflow flow(job_desc, FLAGS_this_machine_name); }
  CloseStdoutAndStderr();
  return 0;
}
