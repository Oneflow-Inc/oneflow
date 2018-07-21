#include "oneflow/core/common/str_util.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/ctrl_server.h"
#include "oneflow/core/job/compiler.h"
#include "oneflow/core/job/improver.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/profiler.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/runtime.h"
#include "oneflow/core/job/available_memory_desc.pb.h"
#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/actor/act_event_logger.h"

namespace oneflow {

namespace {

#define OF_VERSION_MAJOR "0"
#define OF_VERSION_MINOR "1"
#define OF_VERSION_PATCH "0"
#define OF_VERSION OF_VERSION_MAJOR "." OF_VERSION_MINOR "." OF_VERSION_PATCH

std::string BuildVersionString() {
  static const HashMap<std::string, std::string> month_word2num = {
      {"Jan", "01"}, {"Feb", "02"}, {"Mar", "03"}, {"Apr", "04"}, {"May", "05"}, {"Jun", "06"},
      {"Jul", "07"}, {"Aug", "08"}, {"Sep", "09"}, {"Oct", "10"}, {"Nov", "11"}, {"Dec", "12"},
  };
  static const std::string date_str(__DATE__);
  std::string day = date_str.substr(4, 2);
  StringReplace(&day, ' ', '0');
  return OF_VERSION " (" + date_str.substr(7) + month_word2num.at(date_str.substr(0, 3)) + day + "."
         + __TIME__ + ")";
}

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
  Global<CtrlClient>::Get()->PushKV(GetAmdCtrlKey(Global<MachineCtx>::Get()->this_machine_id()),
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
  if (Global<MachineCtx>::Get()->IsThisMachineMaster()) {
    Global<CtrlClient>::Get()->ClearKV("cpu_device_num");
  }
  CHECK_GT(cpu_device_num, 0);
  Global<JobDesc>::Get()->SetCpuDeviceNum(cpu_device_num);
}

}  // namespace

class Oneflow final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Oneflow);
  ~Oneflow() = default;

  Oneflow(const std::string& job_conf_filepath, const std::string& this_mchn_name);

 private:
  std::unique_ptr<CtrlServer> ctrl_server_;
};

Oneflow::Oneflow(const std::string& job_conf_filepath, const std::string& this_mchn_name) {
  // New All Global
  Global<JobDesc>::New(job_conf_filepath);
  Global<MachineCtx>::New(this_mchn_name);
  const MachineCtx* machine_ctx = Global<MachineCtx>::Get();
  if (machine_ctx->IsThisMachineMaster()) { Global<Profiler>::New(); }
  ctrl_server_.reset(new CtrlServer(machine_ctx->GetThisCtrlAddr()));
  Global<CtrlClient>::New();
  FixCpuDeviceNum();
  Global<IDMgr>::New();
  // Compile
  Plan naive_plan;
  Plan plan;
  PushAvailableMemDescOfThisMachine();
  AvailableMemDesc amd;

  if (machine_ctx->IsThisMachineMaster()) {
    naive_plan = Compiler().Compile();
    amd = PullAvailableMemDesc();
    plan = Improver().ImproveMemSharedIdOnly(amd, naive_plan);
    Global<CtrlClient>::Get()->PushKV("mem_shared_plan", plan);
  } else {
    Global<CtrlClient>::Get()->PullKV("mem_shared_plan", &plan);
  }
  OF_BARRIER();
  PrintProtoToTextFile(naive_plan, JoinPath(LogDir(), "naive_plan"));
  PrintProtoToTextFile(plan, JoinPath(LogDir(), "mem_shared_plan"));
  // Experiment Runtime
  { Runtime experiment_run(plan, true); }
  // Improve
  if (machine_ctx->IsThisMachineMaster()) {
    PrintProtoToTextFile(amd, JoinPath(LogDir(), "available_mem_desc"));
    CHECK_GT(amd.machine_amd_size(), 0);
    plan = Improver().Improve(amd, naive_plan,
                              JoinPath(LogDir(), ActEventLogger::experiment_prefix_
                                                     + ActEventLogger::act_event_bin_filename_));
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
  if (machine_ctx->IsThisMachineMaster()) {
    if (Global<JobDesc>::Get()->collect_act_event()) {
      Global<Profiler>::Get()->Profile(plan,
                                       JoinPath(LogDir(), ActEventLogger::act_event_bin_filename_));
    }
  }
  // Delete All Global
  Global<CtrlClient>::Delete();
  ctrl_server_.reset();
  if (machine_ctx->IsThisMachineMaster()) { Global<Profiler>::Delete(); }
  Global<MachineCtx>::Delete();
  Global<IDMgr>::Delete();
  Global<JobDesc>::Delete();
}

}  // namespace oneflow

DEFINE_string(job_conf, "", "");
DEFINE_string(this_machine_name, "", "");

int main(int argc, char** argv) {
  using namespace oneflow;
  google::InitGoogleLogging(argv[0]);
  gflags::SetVersionString(BuildVersionString());
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LocalFS()->RecursivelyCreateDirIfNotExist(LogDir());
  RedirectStdoutAndStderrToGlogDir();
  { Oneflow flow(FLAGS_job_conf, FLAGS_this_machine_name); }
  CloseStdoutAndStderr();
  return 0;
}
