#include "oneflow/core/job/environment_objects_scope.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/control/ctrl_server.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/critical_section_desc.h"
#include "oneflow/core/job/available_memory_desc.pb.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/profiler.h"
#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/job/foreign_job_instance.h"
#include "oneflow/core/job/inter_user_job_info.pb.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/lbi_diff_watcher_info.pb.h"

namespace oneflow {

namespace {

std::string GetAmdCtrlKey(int64_t machine_id) {
  return "AvailableMemDesc/" + std::to_string(machine_id);
}

void PushAvailableMemDescOfThisMachine() {
  AvailableMemDescOfMachine this_machine_mem_desc;
#ifdef WITH_CUDA
  FOR_RANGE(int, i, 0, Global<ResourceDesc>::Get()->GpuDeviceNum()) {
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
  FOR_RANGE(int64_t, i, 0, Global<ResourceDesc>::Get()->TotalMachineNum()) {
    Global<CtrlClient>::Get()->PullKV(GetAmdCtrlKey(i), ret.add_machine_amd());
  }
  return ret;
}

Maybe<void> FixCpuDeviceNum() {
  int32_t cpu_device_num = Global<ResourceDesc>::Get()->CpuDeviceNum();
  if (cpu_device_num > 0) { return Maybe<void>::Ok(); }
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
  OF_CHECK_GT(cpu_device_num, 0);
  Global<ResourceDesc>::Get()->SetCpuDeviceNum(cpu_device_num);
  return Maybe<void>::Ok();
}

}  // namespace

EnvironmentObjectsScope::EnvironmentObjectsScope() {}

Maybe<void> EnvironmentObjectsScope::Init(const ConfigProto& config_proto) {
  flags_and_log_scope_.reset(new FlagsAndLogScope(config_proto, "oneflow"));
  Global<ResourceDesc>::New(config_proto.resource());
  Global<const IOConf>::New(config_proto.io_conf());
  Global<const ProfilerConf>::New(config_proto.profiler_conf());
  Global<CtrlServer>::New();
  Global<CtrlClient>::New();
  OF_BARRIER();
  int64_t this_mchn_id =
      Global<ResourceDesc>::Get()->GetMachineId(Global<CtrlServer>::Get()->this_machine_addr());
  Global<MachineCtx>::New(this_mchn_id);
  JUST(FixCpuDeviceNum());
  Global<IDMgr>::New();
  if (Global<MachineCtx>::Get()->IsThisMachineMaster()
      && Global<const ProfilerConf>::Get()->collect_act_event()) {
    Global<Profiler>::New();
  }
  PushAvailableMemDescOfThisMachine();
  if (Global<MachineCtx>::Get()->IsThisMachineMaster()) {
    Global<AvailableMemDesc>::New();
    *Global<AvailableMemDesc>::Get() = PullAvailableMemDesc();
    Global<CriticalSectionDesc>::New();
    Global<InterUserJobInfo>::New();
    Global<JobBuildAndInferCtxMgr>::New();
    Global<LbiDiffWatcherInfo>::New();
  }
  return Maybe<void>::Ok();
}

EnvironmentObjectsScope::~EnvironmentObjectsScope() {
  if (Global<MachineCtx>::Get()->IsThisMachineMaster()) {
    Global<LbiDiffWatcherInfo>::Delete();
    Global<JobBuildAndInferCtxMgr>::Delete();
    Global<InterUserJobInfo>::Delete();
    Global<CriticalSectionDesc>::Delete();
    Global<AvailableMemDesc>::Delete();
  }
  if (Global<Profiler>::Get() != nullptr) { Global<Profiler>::Delete(); }
  Global<IDMgr>::Delete();
  Global<MachineCtx>::Delete();
  Global<CtrlClient>::Delete();
  Global<CtrlServer>::Delete();
  Global<const ProfilerConf>::Delete();
  Global<const IOConf>::Delete();
  Global<ResourceDesc>::Delete();
  flags_and_log_scope_.reset();
}

}  // namespace oneflow
