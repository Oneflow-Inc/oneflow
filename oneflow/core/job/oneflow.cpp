#include "oneflow/core/common/str_util.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/ctrl_server.h"
#include "oneflow/core/job/compiler.h"
#include "oneflow/core/job/improver.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/profiler.h"
#include "oneflow/core/job/sub_plan.pb.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/runtime.h"
#include "oneflow/core/job/available_memory_desc.pb.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
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

std::string cluster_thrd_ids_key(const std::string& plan_name) {
  return plan_name + "_cluster_thrd_ids";
}

std::string net_topo_key(const std::string& plan_name) { return plan_name + "_net_topo"; }

std::string sub_plan_key(const std::string& plan_name, int64_t machine_id, int64_t thrd_id) {
  return plan_name + "_" + std::to_string(machine_id) + "_" + std::to_string(thrd_id);
}

std::string total_mbn_num_key(const std::string& plan_name) { return plan_name + "_total_mbn_num"; }

void PushPlan(const std::string& plan_name, const Plan& plan) {
  HashMap<int64_t, std::set<int64_t>> machine_id2thrd_id_set;
  HashMap<std::pair<int64_t, int64_t>, std::vector<TaskProto>> mchn_thrd_id2task_protos;
  for (const auto& task : plan.task()) {
    machine_id2thrd_id_set[task.machine_id()].insert(task.thrd_id());
    mchn_thrd_id2task_protos[std::make_pair(task.machine_id(), task.thrd_id())].emplace_back(task);
  }

  HashMap<int64_t, ThrdIds> machine_id2thrd_ids;
  for (const auto& pair : machine_id2thrd_id_set) {
    CHECK(machine_id2thrd_ids.emplace(pair.first, ThrdIds()).second);
    std::vector<int64_t> thrd_id_vec(pair.second.begin(), pair.second.end());
    *(machine_id2thrd_ids.at(pair.first).mutable_thrd_id()) = StdVec2PbRf(thrd_id_vec);
  }

  ClusterThrdIds cluster_thrd_ids;
  *(cluster_thrd_ids.mutable_machine_id2thrd_ids()) = HashMap2PbMap(machine_id2thrd_ids);
  Global<CtrlClient>::Get()->PushKV(cluster_thrd_ids_key(plan_name), cluster_thrd_ids);

  for (const auto& pair : mchn_thrd_id2task_protos) {
    SubPlan sub_plan;
    *(sub_plan.mutable_task()) = StdVec2PbRpf(pair.second);
    Global<CtrlClient>::Get()->PushKV(sub_plan_key(plan_name, pair.first.first, pair.first.second),
                                      sub_plan);
  }
  Global<CtrlClient>::Get()->PushKV(total_mbn_num_key(plan_name),
                                    std::to_string(plan.total_mbn_num()));

  Global<CtrlClient>::Get()->PushKV(net_topo_key(plan_name), plan.net_topo());
}

void PullPlan(const std::string& plan_name, Plan* plan) {
  ClusterThrdIds cluster_thrd_ids;
  Global<CtrlClient>::Get()->PullKV(cluster_thrd_ids_key(plan_name), &cluster_thrd_ids);
  PrintProtoToTextFile(cluster_thrd_ids, JoinPath(FLAGS_log_dir, cluster_thrd_ids_key(plan_name)));
  HashMap<int64_t, ThrdIds> machine_id2thrd_ids;
  machine_id2thrd_ids = PbMap2HashMap(cluster_thrd_ids.machine_id2thrd_ids());
  int64_t machine_id = Global<MachineCtx>::Get()->this_machine_id();
  auto thrd_ids_it = machine_id2thrd_ids.find(machine_id);
  CHECK(thrd_ids_it != machine_id2thrd_ids.end());
  std::vector<int64_t> thrd_id_vec = PbRf2StdVec(thrd_ids_it->second.thrd_id());
  for (auto thrd_id : thrd_id_vec) {
    SubPlan sub_plan;
    Global<CtrlClient>::Get()->PullKV(sub_plan_key(plan_name, machine_id, thrd_id), &sub_plan);
    plan->mutable_task()->MergeFrom(sub_plan.task());
  }
  NetTopo net_topo;
  std::string total_mbn_num;
  Global<CtrlClient>::Get()->PullKV(total_mbn_num_key(plan_name), &total_mbn_num);
  plan->set_total_mbn_num(oneflow_cast<int64_t>(total_mbn_num));
  Global<CtrlClient>::Get()->PullKV(net_topo_key(plan_name), &net_topo);
  *(plan->mutable_net_topo()) = net_topo;
}

#ifdef WITH_CUDA

void EnableCudaPeerAccess() {
  int32_t saved_dev_id;
  CudaCheck(cudaGetDevice(&saved_dev_id));
  int32_t device_count;
  CudaCheck(cudaGetDeviceCount(&device_count));
  FOR_RANGE(int32_t, dev, 0, device_count) {
    CudaCheck(cudaSetDevice(dev));
    FOR_RANGE(int32_t, peer, 0, device_count) {
      int32_t can_access_peer;
      CudaCheck(cudaDeviceCanAccessPeer(&can_access_peer, dev, peer));
      if (can_access_peer == 1) { CudaCheck(cudaDeviceEnablePeerAccess(peer, 0)); }
    }
  }
  CudaCheck(cudaSetDevice(saved_dev_id));
}

void InitialCudaDevices() { EnableCudaPeerAccess(); }

#endif

}  // namespace

class Oneflow final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Oneflow);
  ~Oneflow() = default;

  Oneflow(const std::string& job_conf_filepath);

 private:
  std::unique_ptr<CtrlServer> ctrl_server_;
};

Oneflow::Oneflow(const std::string& job_conf_filepath) {
  // New All Global
  Global<JobDesc>::New(job_conf_filepath);
  ctrl_server_.reset(new CtrlServer());
  Global<CtrlClient>::New();
  OF_BARRIER();
#ifdef WITH_CUDA
  InitialCudaDevices();
#endif
  int64_t this_mchn_id = Global<JobDesc>::Get()->GetMachineId(ctrl_server_->this_machine_addr());
  Global<MachineCtx>::New(this_mchn_id);
  const MachineCtx* machine_ctx = Global<MachineCtx>::Get();
  bool DoProfile =
      machine_ctx->IsThisMachineMaster() && Global<JobDesc>::Get()->collect_act_event();
  if (DoProfile) { Global<Profiler>::New(); }
  FixCpuDeviceNum();
  Global<IDMgr>::New();
  // Compile
  Plan naive_plan;
  Plan mem_shared_plan;
  Plan improved_plan;
  PushAvailableMemDescOfThisMachine();
  AvailableMemDesc amd;
  double start = GetCurTime();

  if (machine_ctx->IsThisMachineMaster()) {
    double start = GetCurTime();
    naive_plan = Compiler().Compile();
    LOG(INFO) << "compile time: " << GetCurTime() - start;
    amd = PullAvailableMemDesc();
    mem_shared_plan = Improver().ImproveMemSharedIdOnly(amd, naive_plan);
    PushPlan("naive_plan", naive_plan);
    PushPlan("mem_shared_plan", mem_shared_plan);
  } else {
    PullPlan("naive_plan", &naive_plan);
    PullPlan("mem_shared_plan", &mem_shared_plan);
  }
  OF_BARRIER();
  TeePersistentLogStream::Create("naive_plan")->Write(naive_plan);
  TeePersistentLogStream::Create("mem_shared_plan")->Write(mem_shared_plan);
  LOG(INFO) << "push_pull_plan:" << GetCurTime() - start;
  if (Global<JobDesc>::Get()->enable_experiment_run()) {
    // Experiment Runtime
    { Runtime experiment_run(mem_shared_plan, true); }
    // Improve
    if (machine_ctx->IsThisMachineMaster()) {
      TeePersistentLogStream::Create("available_mem_desc")->Write(amd);
      CHECK_GT(amd.machine_amd_size(), 0);
      improved_plan = Improver().Improve(
          amd, naive_plan,
          JoinPath(FLAGS_log_dir, ActEventLogger::experiment_act_event_bin_filename()));
      PushPlan("improved_plan", improved_plan);
    } else {
      PullPlan("improved_plan", &improved_plan);
    }
    OF_BARRIER();
    TeePersistentLogStream::Create("improved_plan")->Write(improved_plan);
    Global<CtrlClient>::Get()->Clear();
    OF_BARRIER();
  } else {
    improved_plan = mem_shared_plan;
  }
  // Runtime
  { Runtime run(improved_plan, false); }
  if (DoProfile) {
    Global<Profiler>::Get()->Profile(
        improved_plan, JoinPath(FLAGS_log_dir, ActEventLogger::act_event_bin_filename()));
  }
  // Delete All Global
  Global<CtrlClient>::Delete();
  ctrl_server_.reset();
  Global<Profiler>::Delete();
  Global<MachineCtx>::Delete();
  Global<IDMgr>::Delete();
  Global<JobDesc>::Delete();
}

}  // namespace oneflow

DEFINE_string(job_conf, "", "");

int main(int argc, char** argv) {
  using namespace oneflow;
  FLAGS_log_dir = LogDir();
  google::InitGoogleLogging(argv[0]);
  gflags::SetVersionString(BuildVersionString());
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LocalFS()->RecursivelyCreateDirIfNotExist(FLAGS_log_dir);
  RedirectStdoutAndStderrToGlogDir();
  { Oneflow flow(FLAGS_job_conf); }
  CloseStdoutAndStderr();
  return 0;
}
