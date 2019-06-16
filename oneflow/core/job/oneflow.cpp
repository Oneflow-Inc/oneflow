#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/ctrl_server.h"
#include "oneflow/core/job/compiler.h"
#include "oneflow/core/job/improver.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/profiler.h"
#include "oneflow/core/job/sub_plan.pb.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/runtime.h"
#include "oneflow/core/job/critical_section_desc.h"
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

void FixCpuDeviceNum() {
  int32_t cpu_device_num = Global<ResourceDesc>::Get()->CpuDeviceNum();
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
  Global<ResourceDesc>::Get()->SetCpuDeviceNum(cpu_device_num);
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

void WithJobSetLevelGlobalObjs(
    const std::string& job_set_filepath,
    const std::function<void(const PbRpf<JobConf>& job_confs)>& Handler) {
  // New All Global
  JobSet job_set;
  ParseProtoFromTextFile(job_set_filepath, &job_set);
  Global<JobSet>::New(job_set);
  Global<ResourceDesc>::New(job_set.resource());
  Global<const IOConf>::New(job_set.io_conf());
  Global<const ProfileConf>::New(job_set.profile_conf());
  std::unique_ptr<CtrlServer> ctrl_server(new CtrlServer());
  Global<CtrlClient>::New();
  OF_BARRIER();
  int64_t this_mchn_id =
      Global<ResourceDesc>::Get()->GetMachineId(ctrl_server->this_machine_addr());
  Global<MachineCtx>::New(this_mchn_id);
  FixCpuDeviceNum();
  Global<IDMgr>::New();
  bool DoProfile = Global<MachineCtx>::Get()->IsThisMachineMaster()
                   && Global<const ProfileConf>::Get()->collect_act_event();
  if (DoProfile) { Global<Profiler>::New(); }
  PushAvailableMemDescOfThisMachine();
  if (Global<MachineCtx>::Get()->IsThisMachineMaster()) {
    Global<AvailableMemDesc>::New();
    *Global<AvailableMemDesc>::Get() = PullAvailableMemDesc();
    Global<CriticalSectionDesc>::New();
  }

  Global<std::vector<std::unique_ptr<JobDesc>>>::New();
  FOR_RANGE(int32_t, i, 0, job_set.job_conf_size()) {
    Global<std::vector<std::unique_ptr<JobDesc>>>::Get()->emplace_back(
        new JobDesc(job_set.job_conf(i), i));
  }

  Handler(job_set.job_conf());

  Global<std::vector<std::unique_ptr<JobDesc>>>::Delete();

  if (Global<MachineCtx>::Get()->IsThisMachineMaster()) {
    Global<CriticalSectionDesc>::Delete();
    Global<AvailableMemDesc>::Delete();
  }
  if (DoProfile) { Global<Profiler>::Delete(); }
  Global<IDMgr>::Delete();
  Global<MachineCtx>::Delete();
  Global<CtrlClient>::Delete();
  ctrl_server.reset();
  Global<const ProfileConf>::Delete();
  Global<const IOConf>::Delete();
  Global<ResourceDesc>::Delete();
}

void CompileCurJobOnMaster(Job* job, Plan* improved_plan) {
  const JobDesc* job_desc = Global<JobDesc>::Get();
  Plan naive_plan;
  Plan mem_shared_plan;
  double start = GetCurTime();
  if (Global<MachineCtx>::Get()->IsThisMachineMaster()) {
    Compiler().Compile(job, &naive_plan);
    LOG(INFO) << "compile time: " << GetCurTime() - start;
    mem_shared_plan =
        Improver().ImproveMemSharedIdOnly(*Global<AvailableMemDesc>::Get(), naive_plan);
    OF_BARRIER();
    TeePersistentLogStream::Create("naive_plan")->Write(naive_plan);
    TeePersistentLogStream::Create("mem_shared_plan")->Write(mem_shared_plan);
    LOG(INFO) << "push_pull_plan:" << GetCurTime() - start;
  }
  if (job_desc->enable_experiment_run()) {
    if (Global<MachineCtx>::Get()->IsThisMachineMaster()) {
      PushPlan("mem_shared_plan", mem_shared_plan);
    } else {
      PullPlan("mem_shared_plan", &mem_shared_plan);
    }
    // Experiment Runtime
    { Runtime experiment_run(mem_shared_plan, job_desc->piece_num_of_experiment_phase(), true); }
    // Improve
    if (Global<MachineCtx>::Get()->IsThisMachineMaster()) {
      TeePersistentLogStream::Create("available_mem_desc")->Write(*Global<AvailableMemDesc>::Get());
      CHECK_GT(Global<AvailableMemDesc>::Get()->machine_amd_size(), 0);
      *improved_plan = Improver().Improve(
          *Global<AvailableMemDesc>::Get(), naive_plan,
          JoinPath(FLAGS_log_dir, ActEventLogger::experiment_act_event_bin_filename()));
      OF_BARRIER();
      TeePersistentLogStream::Create("improved_plan")->Write(*improved_plan);
      Global<CtrlClient>::Get()->Clear();
      OF_BARRIER();
    }
  } else {
    *improved_plan = mem_shared_plan;
  }
  LOG(INFO) << "compile and improve time: " << GetCurTime() - start;
}

size_t ComputeTotalPieceNum() {
  const auto& job_descs = *Global<std::vector<std::unique_ptr<JobDesc>>>::Get();
  size_t total_piece_num = 0;
  for (const auto& job_desc : job_descs) {
    total_piece_num = std::max<size_t>(total_piece_num,
                                       job_desc->NumOfPiecesInBatch() * job_desc->TotalBatchNum());
  }
  return total_piece_num;
}

void MergePlan(Plan* plan, const std::vector<Plan>& sub_plans) {
  CHECK(!sub_plans.empty());
  *plan = sub_plans.at(0);
  FOR_RANGE(int32_t, i, 1, sub_plans.size()) {
    plan->mutable_task()->MergeFrom(sub_plans.at(i).task());
    plan->set_total_mbn_num(plan->total_mbn_num() + sub_plans.at(i).total_mbn_num());
  }
  Compiler().GenNetTopo(plan);
}

Job ConvertJobConf2Job(const JobConf& job_conf) {
  Job job;
  *job.mutable_net() = job_conf.net();
  *job.mutable_placement() = job_conf.placement();
  *job.mutable_other() = job_conf.other();
  *job.mutable_arg_op_name() = job_conf.arg_op_name();
  *job.mutable_helper()->mutable_sbp_conf() = job_conf.sbp_conf();
  return job;
}

void GetInterfaceOpBlobInfo(const JobBuilder& job_builder, const std::string& op_name,
                            ParallelConf* parallel_conf, BlobDescProto* blob_desc,
                            SbpParallel* sbp_parallel) {
  std::string obn = "out";
  std::string lbn;
  {
    const auto& op_conf = job_builder.OpConf4OpName(op_name);
    if (op_conf.has_variable_conf()) {
      lbn = op_name + "/" + op_conf.variable_conf().out();
    } else if (op_conf.has_input_conf()) {
      lbn = op_name + "/" + op_conf.input_conf().out();
    } else if (op_conf.has_output_conf()) {
      lbn = op_name + "/" + op_conf.output_conf().out();
    } else {
      UNIMPLEMENTED();
    }
  }
  const auto& helper = job_builder.job().helper();
  *parallel_conf = job_builder.ParallelConf4OpName(op_name);
  *blob_desc = helper.lbn2logical_blob_desc().at(lbn);
  *sbp_parallel =
      helper.sbp_conf().op_name2sbp_signature_conf().at(op_name).bn_in_op2sbp_parallel().at(obn);
}

void CheckJobs(std::vector<Job>* jobs) {
  std::vector<std::unique_ptr<const JobBuilder>> job_builders;
  HashSet<std::string> arg_op_names;
  for (Job& job : *jobs) {
    job_builders.emplace_back(std::make_unique<const JobBuilder>(&job));
    for (const auto& arg_op_name : job.arg_op_name()) { arg_op_names.insert(arg_op_name); }
  }
  HashMap<std::string, HashSet<int64_t>> interface_op_name2job_ids;
  HashSet<std::string> unique_op_name_check;
  FOR_RANGE(int32_t, i, 0, jobs->size()) {
    const auto& job = jobs->at(i);
    for (const auto& op : job.net().op()) {
      if (IsInterfaceOpConf(op)) {
        if (op.has_variable_conf() == false) {
          CHECK(arg_op_names.find(op.name()) != arg_op_names.end());
        }
        CHECK(interface_op_name2job_ids[op.name()].emplace(i).second);
      } else {
        CHECK(unique_op_name_check.find(op.name()) == unique_op_name_check.end());
      }
      unique_op_name_check.emplace(op.name());
    }
  }
  for (const auto& pair : interface_op_name2job_ids) {
    if (pair.second.size() <= 1) { continue; }
    bool op_as_output_found = false;
    for (int64_t job_id : pair.second) {
      if (job_builders.at(job_id)->OpConf4OpName(pair.first).has_output_conf()) {
        CHECK_EQ(op_as_output_found, false);
        op_as_output_found = true;
      }
    }
    ParallelConf first_op_parallel_conf;
    BlobDescProto first_op_out_blob_desc;
    SbpParallel first_op_out_blob_sbp;
    GetInterfaceOpBlobInfo(*job_builders.at(*pair.second.begin()), pair.first,
                           &first_op_parallel_conf, &first_op_out_blob_desc,
                           &first_op_out_blob_sbp);
    for (int64_t job_id : pair.second) {
      ParallelConf parallel_conf;
      BlobDescProto out_blob_desc;
      SbpParallel out_blob_sbp;
      GetInterfaceOpBlobInfo(*job_builders.at(job_id), pair.first, &parallel_conf, &out_blob_desc,
                             &out_blob_sbp);
      CHECK(first_op_parallel_conf == parallel_conf);
      CHECK(BlobDesc(first_op_out_blob_desc) == BlobDesc(out_blob_desc));
      CHECK(first_op_out_blob_sbp == out_blob_sbp);
    }
  }
}

void CompileAndMergePlanOnMaster(const PbRpf<JobConf>& job_confs, Plan* plan) {
  std::vector<Job> jobs(job_confs.size());
  std::vector<Plan> sub_plans(job_confs.size());
  FOR_RANGE(int32_t, i, 0, sub_plans.size()) {
    Global<JobDesc>::New(job_confs.Get(i), i);
    jobs.at(i) = ConvertJobConf2Job(job_confs.Get(i));
    CompileCurJobOnMaster(&jobs.at(i), &sub_plans.at(i));
    Global<JobDesc>::Delete();
  }
  CheckJobs(&jobs);
  MergePlan(plan, sub_plans);
}

}  // namespace

class Oneflow final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Oneflow);
  ~Oneflow() = default;

  Oneflow(const std::string& job_set_filepath);
};

Oneflow::Oneflow(const std::string& job_set_filepath) {
  WithJobSetLevelGlobalObjs(job_set_filepath, [&](const PbRpf<JobConf>& job_confs) {
    // Runtime
    Plan plan;
    CompileAndMergePlanOnMaster(job_confs, &plan);
    if (Global<MachineCtx>::Get()->IsThisMachineMaster()) {
      PushPlan("plan", plan);
    } else {
      PullPlan("plan", &plan);
    }
    { Runtime run(plan, ComputeTotalPieceNum(), false); }
    if (Global<Profiler>::Get() != nullptr) {
      Global<Profiler>::Get()->Profile(
          plan, JoinPath(FLAGS_log_dir, ActEventLogger::act_event_bin_filename()));
    }
  });
}

}  // namespace oneflow

DEFINE_string(job_set, "", "");

int main(int argc, char** argv) {
  using namespace oneflow;
  FLAGS_log_dir = LogDir();
  google::InitGoogleLogging(argv[0]);
  gflags::SetVersionString(BuildVersionString());
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LocalFS()->RecursivelyCreateDirIfNotExist(FLAGS_log_dir);
  RedirectStdoutAndStderrToGlogDir();
  { Oneflow flow(FLAGS_job_set); }
  CloseStdoutAndStderr();
  return 0;
}
