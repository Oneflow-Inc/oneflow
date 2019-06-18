#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/buffer_manager.h"
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
#include "oneflow/core/graph/plan_task_graph.h"

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
  Global<BufferMgr<int32_t>>::New();
  Global<BufferMgr<int32_t>>::Get()->NewChannel(kChannelNameGlobalWaitJobId,
                                                job_set.job_conf_size());

  Handler(job_set.job_conf());

  Global<BufferMgr<int32_t>>::Delete();
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

void CompileCurJobOnMaster(Job* job, Plan* improved_plan, bool need_job_complete) {
  const JobDesc* job_desc = Global<JobDesc>::Get();
  Plan naive_plan;
  Plan mem_shared_plan;
  double start = GetCurTime();
  if (Global<MachineCtx>::Get()->IsThisMachineMaster()) {
    Compiler().Compile(job, &naive_plan, need_job_complete);
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

RegstDescProto* GetSoleDataRegstDescProto(TaskProto* task) {
  RegstDescProto* ret = nullptr;
  for (auto& pair : *task->mutable_produced_regst_desc()) {
    CHECK(pair.second.regst_desc_type().has_data_regst_desc());
    CHECK_ISNULL(ret);
    ret = &pair.second;
  }
  CHECK_NOTNULL(ret);
  return ret;
}

const OperatorConf& GetSoleOpConf(const TaskProto& task) {
  CHECK_EQ(task.exec_sequence().exec_node_size(), 1);
  return task.exec_sequence().exec_node(0).kernel_conf().op_attribute().op_conf();
}

LogicalBlobId* GetDataRegstSoleLbi(RegstDescProto* regst) {
  CHECK(regst->regst_desc_type().has_data_regst_desc());
  auto* data_regst_desc = regst->mutable_regst_desc_type()->mutable_data_regst_desc();
  return data_regst_desc->mutable_lbi2blob_desc(0)->mutable_lbi();
}

void UpdateSoleObnRegstDescId(TaskProto* task) {
  CHECK_EQ(task->exec_sequence().exec_node_size(), 1);
  auto* exec_node = task->mutable_exec_sequence()->mutable_exec_node(0);
  const auto& obns = exec_node->kernel_conf().op_attribute().output_bns();
  CHECK_EQ(obns.size(), 1);
  int64_t regst_desc_id = GetSoleDataRegstDescProto(task)->regst_desc_id();
  (*exec_node->mutable_bn_in_op2regst_desc_id())[obns.Get(0)] = regst_desc_id;
}

// example
// given caller plan: op_A --> op_identity_tick --> op_B
// given callee plan: op_src_tick --> op_C --> op_D --> op_E --> op_sink_tick
// return:
//         op_A --> op_identity_tick --> op_C --> op_D --> op_E --> op_sink_tick --> op_B
//                                        /
//                        op_src_tick ---/
//
// note: after this function called, op_src_tick is illegal and need to be deleted from plan
void LinkTickTaskProto(TaskProto* identity_tick, TaskProto* src_tick, TaskProto* sink_tick) {
  CHECK(GetSoleOpConf(*identity_tick).has_tick_conf());
  CHECK(GetSoleOpConf(*src_tick).has_source_tick_conf());
  CHECK(GetSoleOpConf(*sink_tick).has_sink_tick_conf());
  RegstDescProto* id_tick_sole_regst = GetSoleDataRegstDescProto(identity_tick);
  RegstDescProto* sink_tick_sole_regst = GetSoleDataRegstDescProto(sink_tick);
  RegstDescProto id_tick_sole_regst_value = *id_tick_sole_regst;

  *id_tick_sole_regst = *GetSoleDataRegstDescProto(src_tick);
  *GetDataRegstSoleLbi(id_tick_sole_regst) = *GetDataRegstSoleLbi(&id_tick_sole_regst_value);

  *GetDataRegstSoleLbi(&id_tick_sole_regst_value) = *GetDataRegstSoleLbi(sink_tick_sole_regst);
  *sink_tick_sole_regst = id_tick_sole_regst_value;

  UpdateSoleObnRegstDescId(identity_tick);
  UpdateSoleObnRegstDescId(sink_tick);
}

void LinkMainPlan(Plan* plan, const Plan& main_plan,
                  const std::vector<std::string>& identity_tick_op_names) {
  plan->mutable_task()->MergeFrom(main_plan.task());
  HashMap<std::string, TaskProto*> sole_op_name2sole_task;
  FOR_RANGE(int64_t, i, 0, plan->task_size()) {
    TaskProto* task = plan->mutable_task(i);
    if (task->exec_sequence().exec_node_size() == 1) {
      const auto& kernel_conf = task->exec_sequence().exec_node(0).kernel_conf();
      const auto& op_name = kernel_conf.op_attribute().op_conf().name();
      CHECK(sole_op_name2sole_task.emplace(op_name, task).second);
    }
  }
  FOR_RANGE(int32_t, i, 0, Global<CriticalSectionDesc>::Get()->CriticalSectionNum()) {
    const CriticalSectionId& critical_section_id =
        Global<CriticalSectionDesc>::Get()->GetCriticalSectionByIndex(i).critical_section_id();
    LinkTickTaskProto(sole_op_name2sole_task.at(identity_tick_op_names.at(i)),
                      sole_op_name2sole_task.at(critical_section_id.source_tick_op_name()),
                      sole_op_name2sole_task.at(critical_section_id.sink_tick_op_name()));
  }
  {
    // erase source_tick task_proto
    HashSet<std::string> source_tick_op_names;
    FOR_RANGE(int32_t, i, 0, Global<CriticalSectionDesc>::Get()->CriticalSectionNum()) {
      const CriticalSectionId& critical_section_id =
          Global<CriticalSectionDesc>::Get()->GetCriticalSectionByIndex(i).critical_section_id();
      CHECK(source_tick_op_names.emplace(critical_section_id.source_tick_op_name()).second);
    }
    Erase<PbRpf<TaskProto>>(*plan->mutable_task(), [&](const TaskProto& task) {
      if (task.task_type() == TaskType::kSourceTick) {
        CHECK(task.exec_sequence().exec_node_size() == 1);
        const auto& kernel_conf = task.exec_sequence().exec_node(0).kernel_conf();
        const auto& op_conf = kernel_conf.op_attribute().op_conf();
        CHECK(op_conf.has_source_tick_conf());
        CHECK(source_tick_op_names.find(op_conf.name()) != source_tick_op_names.end());
        return true;
      } else {
        return false;
      }
    });
  }
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

JobConf ConvertJob2JobConf(const Job& job) {
  JobConf job_conf;
  *job_conf.mutable_net() = job.net();
  *job_conf.mutable_placement() = job.placement();
  *job_conf.mutable_other() = job.other();
  *job_conf.mutable_arg_op_name() = job.arg_op_name();
  *job_conf.mutable_sbp_conf() = job.helper().sbp_conf();
  return job_conf;
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

HashSet<std::string> GetArgOpNames(const std::vector<Job>& jobs) {
  HashSet<std::string> arg_op_names;
  for (const Job& job : jobs) {
    for (const auto& arg_op_name : job.arg_op_name()) { arg_op_names.insert(arg_op_name); }
  }
  return arg_op_names;
}

HashMap<std::string, HashSet<int64_t>> GetInterfaceOpName2JobIds(const std::vector<Job>& jobs) {
  HashSet<std::string> arg_op_names = GetArgOpNames(jobs);
  HashMap<std::string, HashSet<int64_t>> interface_op_name2job_ids;
  HashSet<std::string> unique_op_name_check;
  FOR_RANGE(int32_t, i, 0, jobs.size()) {
    const auto& job = jobs.at(i);
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
  return interface_op_name2job_ids;
}

void CheckJobs(std::vector<Job>* jobs) {
  std::vector<std::unique_ptr<const JobBuilder>> job_builders;
  for (Job& job : *jobs) { job_builders.emplace_back(std::make_unique<const JobBuilder>(&job)); }
  for (const auto& pair : GetInterfaceOpName2JobIds(*jobs)) {
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

std::vector<TaskProto*> SortSameOpNameTaskProtos(const std::string& op_name, Plan* plan) {
  std::vector<TaskProto*> task_protos;
  FOR_RANGE(int64_t, i, 0, plan->task_size()) {
    TaskProto* task = plan->mutable_task(i);
    if (task->exec_sequence().exec_node_size() == 1) {
      const KernelConf& kernel_conf = task->exec_sequence().exec_node(0).kernel_conf();
      if (op_name == kernel_conf.op_attribute().op_conf().name()) {
        task_protos.emplace_back(task);
      }
    }
  }
  std::sort(task_protos.begin(), task_protos.end(), [](const TaskProto* lhs, const TaskProto* rhs) {
    return lhs->machine_id() < rhs->machine_id()
           || (lhs->machine_id() == rhs->machine_id() && lhs->thrd_id() < rhs->thrd_id());
  });
  return task_protos;
}

RegstDescProto* GetSoleDataRegst(TaskProto* task_proto) {
  RegstDescProto* ret = nullptr;
  for (auto& pair : *task_proto->mutable_produced_regst_desc()) {
    RegstDescProto* regst_desc = &pair.second;
    if (regst_desc->regst_desc_type().has_data_regst_desc()) {
      CHECK_ISNULL(ret);
      CHECK_EQ(regst_desc->regst_desc_type().data_regst_desc().lbi2blob_desc_size(), 1);
      ret = regst_desc;
    }
  }
  CHECK_NOTNULL(ret);
  return ret;
}

void BindInterfaceMemBlobId(const std::vector<Job>& jobs, std::vector<Plan>* sub_plans) {
  for (const auto& pair : GetInterfaceOpName2JobIds(jobs)) {
    std::vector<std::vector<TaskProto*>> same_op_name_sorted_task_protos;
    for (int64_t job_id : pair.second) {
      same_op_name_sorted_task_protos.push_back(
          SortSameOpNameTaskProtos(pair.first, &sub_plans->at(job_id)));
    }
    const auto& first_vec = same_op_name_sorted_task_protos.at(0);
    for (const auto& task_protos : same_op_name_sorted_task_protos) {
      CHECK_EQ(task_protos.size(), first_vec.size());
      FOR_RANGE(int32_t, i, 0, first_vec.size()) {
        CHECK_EQ(task_protos.at(i)->machine_id(), first_vec.at(i)->machine_id());
        CHECK_EQ(task_protos.at(i)->thrd_id(), first_vec.at(i)->thrd_id());
        const RegstDescProto& first_regst_desc = *GetSoleDataRegst(first_vec.at(i));
        CHECK_EQ(first_regst_desc.mem_shared_offset(), 0);
        RegstDescProto* regst_desc = GetSoleDataRegst(task_protos.at(i));
        CHECK_EQ(regst_desc->mem_shared_offset(), 0);
        regst_desc->set_mem_shared_id(first_regst_desc.mem_shared_id());
      }
    }
  }
}

void MakeMainJob(const std::vector<Job>& jobs, Job* main_job,
                 std::vector<std::string>* identity_tick_op_names) {
  std::vector<OperatorConf> op_confs;
  op_confs.push_back(OperatorConf());
  OperatorConf& wait_and_send_ids_op_conf = op_confs.back();
  {
    wait_and_send_ids_op_conf.set_name(std::string("System-Main-WaitAndSendIds_") + NewUniqueId());
    auto* wait_and_send_ids_conf = wait_and_send_ids_op_conf.mutable_wait_and_send_ids_conf();
    wait_and_send_ids_conf->set_out("out");
    wait_and_send_ids_conf->set_wait_channel_name(kChannelNameGlobalWaitJobId);
    FOR_RANGE(int64_t, i, 0, Global<std::vector<std::unique_ptr<JobDesc>>>::Get()->size() - 1) {
      const auto& cs_idx = Global<CriticalSectionDesc>::Get()->CriticalSectionIndexes4JobId(i);
      *wait_and_send_ids_conf->add_id_list()->mutable_id() = {cs_idx.begin(), cs_idx.end()};
    }
  }
  op_confs.push_back(OperatorConf());
  OperatorConf& cs_case_op_conf = op_confs.back();
  {
    cs_case_op_conf.set_name(std::string("System-Main-Case_") + NewUniqueId());
    auto* cs_case_conf = cs_case_op_conf.mutable_case_conf();
    cs_case_conf->set_in(wait_and_send_ids_op_conf.name() + "/out");
    FOR_RANGE(int64_t, i, 0, Global<CriticalSectionDesc>::Get()->CriticalSectionNum()) {
      cs_case_conf->add_out(GenRepeatedBn("out", i));
    }
  }
  FOR_RANGE(int64_t, i, 0, Global<CriticalSectionDesc>::Get()->CriticalSectionNum()) {
    op_confs.push_back(OperatorConf());
    OperatorConf& identity_tick_op_conf = op_confs.back();
    identity_tick_op_conf.set_name(std::string("System-Main-Tick_") + NewUniqueId());
    auto* identity_tick_conf = identity_tick_op_conf.mutable_tick_conf();
    identity_tick_conf->add_tick(cs_case_op_conf.name() + "/" + GenRepeatedBn("out", i));
    identity_tick_conf->set_out("out");
    identity_tick_op_names->push_back(identity_tick_op_conf.name());
  }
  op_confs.push_back(OperatorConf());
  OperatorConf& cs_esac_op_conf = op_confs.back();
  {
    cs_esac_op_conf.set_name(std::string("System-Main-Esac_") + NewUniqueId());
    auto* cs_esac_conf = cs_esac_op_conf.mutable_esac_conf();
    for (const auto& identity_tick_op_name : *identity_tick_op_names) {
      cs_esac_conf->add_in(identity_tick_op_name + "/out");
    }
    cs_esac_conf->set_out("out");
  }

  ParallelConf parallel_conf;
  parallel_conf.set_policy(kDataParallel);
  parallel_conf.add_device_name("0:cpu:0");
  JobBuilder(main_job).AddOps(parallel_conf, op_confs);
  main_job->mutable_other()->mutable_predict_conf();
  main_job->mutable_other()->set_piece_size(1);
  main_job->mutable_other()->set_data_part_num(1);
  main_job->mutable_other()->set_total_batch_num(1);
  main_job->mutable_other()->set_default_data_type(DataType::kInt32);
}

void CompileMainJob(Job* main_job, int32_t job_id, Plan* main_plan) {
  JobConf job_conf = ConvertJob2JobConf(*main_job);
  Global<JobDesc>::New(job_conf, job_id);
  CompileCurJobOnMaster(main_job, main_plan, false);
  Global<JobDesc>::Delete();
}

void AddGlobalJobDesc(const Job& job, int32_t job_id) {
  JobConf job_conf = ConvertJob2JobConf(job);
  auto* job_descs = Global<std::vector<std::unique_ptr<JobDesc>>>::Get();
  CHECK_EQ(job_descs->size(), job_id);
  job_descs->emplace_back(new JobDesc(job_conf, job_id));
}

void FinishGlobalCriticalSectionDesc(const std::vector<Plan>& plans) {
  std::vector<std::unique_ptr<PlanTaskGraph>> plan_task_graphs;
  for (const auto& plan : plans) { plan_task_graphs.emplace_back(new PlanTaskGraph(plan)); }
  HashSet<int64_t> input_output_mem_block_ids;
  auto* critical_section_desc = Global<CriticalSectionDesc>::Get();
  FOR_RANGE(int64_t, i, 0, critical_section_desc->CriticalSectionNum()) {
    auto* critical_section = critical_section_desc->MutCriticalSectionByIndex(i);
    if (critical_section->critical_section_type() == kInputCriticalSection) {
      TODO();
    } else if (critical_section->critical_section_type() == kOutputCriticalSection) {
      TODO();
    } else {
      CHECK_EQ(critical_section->critical_section_type(), kTotalJobCriticalSection);
    }
  }
  FOR_RANGE(int64_t, i, 0, critical_section_desc->CriticalSectionNum()) {
    auto* critical_section = critical_section_desc->MutCriticalSectionByIndex(i);
    if (critical_section->critical_section_type() == kTotalJobCriticalSection) { TODO(); }
  }
  critical_section_desc->Done();
}

void CompileAndMergePlanOnMaster(const PbRpf<JobConf>& job_confs, Plan* plan) {
  std::vector<Job> jobs(job_confs.size());
  std::vector<Plan> sub_plans(job_confs.size());
  FOR_RANGE(int32_t, i, 0, sub_plans.size()) {
    Global<JobDesc>::New(job_confs.Get(i), i);
    jobs.at(i) = ConvertJobConf2Job(job_confs.Get(i));
    CompileCurJobOnMaster(&jobs.at(i), &sub_plans.at(i), true);
    Global<JobDesc>::Delete();
  }
  if (Global<MachineCtx>::Get()->IsThisMachineMaster()) {
    CheckJobs(&jobs);
    BindInterfaceMemBlobId(jobs, &sub_plans);
    FinishGlobalCriticalSectionDesc(sub_plans);
    MergePlan(plan, sub_plans);
    Plan main_plan;
    std::vector<std::string> identity_tick_op_names;
    {
      Job main_job;
      MakeMainJob(jobs, &main_job, &identity_tick_op_names);
      AddGlobalJobDesc(main_job, sub_plans.size());
      CompileMainJob(&main_job, sub_plans.size(), &main_plan);
    }
    LinkMainPlan(plan, main_plan, identity_tick_op_names);

    PushPlan("merged_plan", *plan);
    TeePersistentLogStream::Create("merged_plan")->Write(*plan);
  } else {
    PullPlan("merged_plan", plan);
  }
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
