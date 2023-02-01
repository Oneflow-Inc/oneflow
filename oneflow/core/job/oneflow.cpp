/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/common/constant.h"
#include "oneflow/core/common/range.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/job/compiler.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/sub_plan.pb.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/job/oneflow.h"
#include "oneflow/core/job/inter_job_mem_sharing_util.h"
#include "oneflow/core/job/plan_util.h"
#include "oneflow/core/operator/interface_op_util.h"
#include "oneflow/core/job/critical_section_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/vm/virtual_machine.h"
#include "oneflow/core/graph/plan_task_graph.h"
#include "oneflow/core/graph/boxing/collective_boxing_util.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/job_rewriter/job_completer.h"

namespace std {

template<>
struct hash<oneflow::ParallelBlobConf> {
  size_t operator()(const oneflow::ParallelBlobConf& parallel_blob_conf) const {
    std::string serialized;
    parallel_blob_conf.SerializeToString(&serialized);
    return std::hash<std::string>()(serialized);
  }
};

}  // namespace std

namespace oneflow {

bool operator==(const ParallelBlobConf& lhs, const ParallelBlobConf& rhs) {
  return BlobDesc(lhs.logical_blob_desc_conf()) == BlobDesc(rhs.logical_blob_desc_conf())
         && lhs.parallel_conf() == rhs.parallel_conf() && lhs.nd_sbp() == rhs.nd_sbp();
}

namespace {

// There are circles in MainJob.
// A MainJob is a Job like:
//
// wait_and_send_ids_op -> reentrant_lock_op -> case_op -> identity_op -> esac_op ->
//                                \________________________________________________/
//
// back edges esac_op -> reentrant_lock_op are linked by rewriting the plan instead of
// compiling OpGraph to TaskGraph.
// ReentrantLockBackEdge holds the key information of a back edge
struct ReentrantLockBackEdge {
  std::string reentrant_lock_op_name;       // back edge destination.
  LogicalBlobId critical_section_sink_lbi;  // back edge source.
};

std::string cluster_thrd_ids_key(const std::string& plan_name) {
  return plan_name + "_cluster_thrd_ids";
}

std::string ctrl_regst_desc_info_key(const std::string& plan_name) {
  return plan_name + "_ctrl_regst_desc_info_key";
}

std::string job_id2job_conf(const std::string& plan_name) { return plan_name + "_job_id2job_conf"; }

std::string GetCollectiveBoxingPlanKey(const std::string& plan_name) {
  return plan_name + "_collective_boxing_plan";
}

std::string sub_plan_key(const std::string& plan_name, int64_t machine_id, int64_t thrd_id) {
  return plan_name + "_" + std::to_string(machine_id) + "_" + std::to_string(thrd_id);
}

std::string block7chunk_key(const std::string& plan_name, int64_t machine_id) {
  return plan_name + "_" + std::to_string(machine_id) + "_block7chunk";
}

void PushPlan(const std::string& plan_name, Plan&& plan) {
  HashMap<int64_t, std::set<int64_t>> machine_id2thrd_id_set;
  HashMap<std::pair<int64_t, int64_t>, std::list<TaskProto>> mchn_thrd_id2task_protos;
  HashMap<int64_t, MemBlockAndChunkList> machine_id2block7chunk;

  for (TaskProto& task : *plan.mutable_task()) {
    machine_id2thrd_id_set[task.machine_id()].insert(task.thrd_id());
    mchn_thrd_id2task_protos[std::make_pair(task.machine_id(), task.thrd_id())].emplace_back(
        std::move(task));
  }

  HashMap<int64_t, ThrdIds> machine_id2thrd_ids;
  for (const auto& pair : machine_id2thrd_id_set) {
    CHECK(machine_id2thrd_ids.emplace(pair.first, ThrdIds()).second);
    std::vector<int64_t> thrd_id_vec(pair.second.begin(), pair.second.end());
    *(machine_id2thrd_ids.at(pair.first).mutable_thrd_id()) = StdVec2PbRf(thrd_id_vec);
  }

  ClusterThrdIds cluster_thrd_ids;
  *(cluster_thrd_ids.mutable_machine_id2thrd_ids()) = HashMap2PbMap(machine_id2thrd_ids);
  Singleton<CtrlClient>::Get()->PushKV(cluster_thrd_ids_key(plan_name), cluster_thrd_ids);

  for (std::pair<const std::pair<int64_t, int64_t>, std::list<oneflow::TaskProto>>& pair :
       mchn_thrd_id2task_protos) {
    SubPlan sub_plan;
    sub_plan.mutable_task()->Reserve(pair.second.size());
    while (!pair.second.empty()) {
      sub_plan.mutable_task()->Add(std::move(pair.second.front()));
      pair.second.pop_front();
    }
    Singleton<CtrlClient>::Get()->PushKV(
        sub_plan_key(plan_name, pair.first.first, pair.first.second), sub_plan);
  }

  for (const auto& mem_block : plan.block_chunk_list().mem_block()) {
    *machine_id2block7chunk[mem_block.machine_id()].add_mem_block() = mem_block;
  }
  for (const auto& chunk : plan.block_chunk_list().chunk()) {
    *machine_id2block7chunk[chunk.machine_id()].add_chunk() = chunk;
  }
  for (const auto& pair : machine_id2block7chunk) {
    Singleton<CtrlClient>::Get()->PushKV(block7chunk_key(plan_name, pair.first), pair.second);
  }

  Singleton<CtrlClient>::Get()->PushKV(ctrl_regst_desc_info_key(plan_name),
                                       plan.ctrl_regst_desc_info());
  Singleton<CtrlClient>::Get()->PushKV(job_id2job_conf(plan_name), plan.job_confs());
  Singleton<CtrlClient>::Get()->PushKV(GetCollectiveBoxingPlanKey(plan_name),
                                       plan.collective_boxing_plan());
}

void PullPlan(const std::string& plan_name, Plan* plan) {
  ClusterThrdIds cluster_thrd_ids;
  Singleton<CtrlClient>::Get()->PullKV(cluster_thrd_ids_key(plan_name), &cluster_thrd_ids);
  PrintProtoToTextFile(cluster_thrd_ids, JoinPath(FLAGS_log_dir, cluster_thrd_ids_key(plan_name)));
  HashMap<int64_t, ThrdIds> machine_id2thrd_ids;
  machine_id2thrd_ids = PbMap2HashMap(cluster_thrd_ids.machine_id2thrd_ids());
  int64_t machine_id = GlobalProcessCtx::Rank();
  auto thrd_ids_it = machine_id2thrd_ids.find(machine_id);
  CHECK(thrd_ids_it != machine_id2thrd_ids.end());
  std::vector<int64_t> thrd_id_vec = PbRf2StdVec(thrd_ids_it->second.thrd_id());
  for (auto thrd_id : thrd_id_vec) {
    SubPlan sub_plan;
    Singleton<CtrlClient>::Get()->PullKV(sub_plan_key(plan_name, machine_id, thrd_id), &sub_plan);
    plan->mutable_task()->MergeFrom(sub_plan.task());
  }
  CtrlRegstDescInfo ctrl_regst_desc_info;
  Singleton<CtrlClient>::Get()->PullKV(ctrl_regst_desc_info_key(plan_name), &ctrl_regst_desc_info);
  *(plan->mutable_ctrl_regst_desc_info()) = ctrl_regst_desc_info;
  JobConfs job_confs;
  Singleton<CtrlClient>::Get()->PullKV(job_id2job_conf(plan_name), &job_confs);
  *(plan->mutable_job_confs()) = job_confs;
  Singleton<CtrlClient>::Get()->PullKV(GetCollectiveBoxingPlanKey(plan_name),
                                       plan->mutable_collective_boxing_plan());
  MemBlockAndChunkList block7chunk;
  Singleton<CtrlClient>::Get()->PullKV(block7chunk_key(plan_name, machine_id), &block7chunk);
  plan->mutable_block_chunk_list()->CopyFrom(block7chunk);
  // pull op_attribute_info
  OpAttributeInfo op_attribute_info;
  Singleton<CtrlClient>::Get()->PullKV("op_attribute_info", &op_attribute_info);
  // populate op_attribute_info
  PlanUtil::PopulateOpAttribute(plan, op_attribute_info.job_id2op_attribute_ref_table());
}

Maybe<void> CompileCurJobOnMaster(Job* job, Plan* plan, bool need_job_complete) {
  const JobDesc& job_desc = GlobalJobDesc();
  if (GlobalProcessCtx::IsThisProcessMaster()) {
    double start = GetCurTime();
    if (need_job_complete) { JUST(JobCompleter::Complete(job)); }
    Compiler().Compile(job, plan);
    PlanUtil::GenMemBlockAndChunk4Plan(plan);

    LOG(INFO) << "\njob_id: " << job_desc.job_id() << " , job_name: " << job_desc.job_name()
              << " , compile time: " << (GetCurTime() - start) / 1000000000.0 << " seconds.\n";
    if (Singleton<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
      TeePersistentLogStream::Create(StrCat("subplan_job_", job_desc.job_id()))->Write(*plan);
    }
  }
  PlanUtil::GenCollectiveBoxingPlan(job, plan);
  PlanUtil::GenRegisterHint(plan);
  return Maybe<void>::Ok();
}

void MergePlan(Plan* plan, Plan&& other) {
  PbRpf<TaskProto>* dst_tasks = plan->mutable_task();
  PbRpf<TaskProto>* src_tasks = other.mutable_task();
  dst_tasks->Reserve(dst_tasks->size() + src_tasks->size());
  for (TaskProto& task : *src_tasks) { *(dst_tasks->Add()) = std::move(task); }
  plan->mutable_block_chunk_list()->MergeFrom(other.block_chunk_list());

  for (const auto& pair : other.job_confs().job_id2job_conf()) {
    CHECK(plan->mutable_job_confs()->mutable_job_id2job_conf()->insert(pair).second);
  }
  for (const auto& pair : other.collective_boxing_plan().job_id2request_set()) {
    CHECK(
        plan->mutable_collective_boxing_plan()->mutable_job_id2request_set()->insert(pair).second);
  }
  for (auto& pair : *(other.mutable_job_id2op_attribute_ref_table())) {
    CHECK(plan->job_id2op_attribute_ref_table().find(pair.first)
          == plan->job_id2op_attribute_ref_table().end())
        << "fail to merge op attribute info for job: " << pair.first;
    (*plan->mutable_job_id2op_attribute_ref_table())[pair.first] = std::move(pair.second);
  }
}

void MergeSubPlan(Plan* plan, std::vector<Plan>&& sub_plans) {
  CHECK(!sub_plans.empty());
  *plan = std::move(sub_plans.at(0));
  FOR_RANGE(int32_t, i, 1, sub_plans.size()) { MergePlan(plan, std::move(sub_plans.at(i))); }
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

const OperatorConf& GetSoleOpConf(Plan* plan, const TaskProto& task) {
  CHECK_EQ(task.exec_sequence().exec_node_size(), 1);
  return PlanUtil::GetOpAttribute(plan, task.job_id(),
                                  task.exec_sequence().exec_node(0).kernel_conf())
      .op_conf();
}

void UpdateSoleObnRegstDescId(Plan* plan, TaskProto* task) {
  CHECK_EQ(task->exec_sequence().exec_node_size(), 1);
  auto* exec_node = task->mutable_exec_sequence()->mutable_exec_node(0);
  const auto& obns =
      PlanUtil::GetOpAttribute(plan, task->job_id(), exec_node->kernel_conf()).output_bns();
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
//                        op_src_tick -->/
//
// note: after this function called, op_src_tick is illegal and need to be deleted from plan
void LinkTickTaskProto(Plan* plan, TaskProto* identity_tick, TaskProto* src_tick,
                       TaskProto* sink_tick) {
  CHECK(GetSoleOpConf(plan, *identity_tick).has_tick_conf());
  CHECK(GetSoleOpConf(plan, *src_tick).has_source_tick_conf());
  CHECK(GetSoleOpConf(plan, *sink_tick).has_sink_tick_conf());
  RegstDescProto* id_tick_sole_regst = GetSoleDataRegstDescProto(identity_tick);
  RegstDescProto* src_tick_sole_regst = GetSoleDataRegstDescProto(src_tick);
  RegstDescProto* sink_tick_sole_regst = GetSoleDataRegstDescProto(sink_tick);

  sink_tick_sole_regst->set_regst_desc_id(id_tick_sole_regst->regst_desc_id());
  *sink_tick_sole_regst->mutable_consumer_task_id() = id_tick_sole_regst->consumer_task_id();
  UpdateSoleObnRegstDescId(plan, sink_tick);
  CHECK_EQ(identity_tick->machine_id(), sink_tick->machine_id());

  id_tick_sole_regst->set_regst_desc_id(src_tick_sole_regst->regst_desc_id());
  *id_tick_sole_regst->mutable_consumer_task_id() = src_tick_sole_regst->consumer_task_id();
  UpdateSoleObnRegstDescId(plan, identity_tick);
}

void LinkMainPlan(Plan* plan, Plan&& main_plan,
                  const std::vector<std::map<int64_t, std::string>>& identity_tick_op_names) {
  std::function<bool(const TaskProto*)> IsInterfaceTickTockTask;
  {
    auto task_ids = std::make_shared<HashSet<int64_t>>();
    for (const auto& task : main_plan.task()) {
      if (task.task_type() == TaskType::kTick) { CHECK(task_ids->emplace(task.task_id()).second); }
    }
    IsInterfaceTickTockTask = [task_ids, plan](const TaskProto* task) {
      if (task_ids->find(task->task_id()) != task_ids->end()) { return true; }
      if (task->exec_sequence().exec_node_size() != 1) { return false; }
      const auto& kernel_conf = task->exec_sequence().exec_node(0).kernel_conf();
      OperatorConf::OpTypeCase op_type_case =
          PlanUtil::GetOpAttribute(plan, task->job_id(), kernel_conf).op_conf().op_type_case();
      return op_type_case == OperatorConf::kSourceTickConf
             || op_type_case == OperatorConf::kSinkTickConf;
    };
  }
  MergePlan(plan, std::move(main_plan));
  HashMap<std::string, TaskProto*> sole_tick_op_name2sole_task;
  FOR_RANGE(int64_t, i, 0, plan->task_size()) {
    TaskProto* task = plan->mutable_task(i);
    if (IsInterfaceTickTockTask(task) == false) { continue; }
    const auto& kernel_conf = task->exec_sequence().exec_node(0).kernel_conf();
    const auto& op_name =
        PlanUtil::GetOpAttribute(plan, task->job_id(), kernel_conf).op_conf().name();
    CHECK(sole_tick_op_name2sole_task.emplace(op_name, task).second);
  }
  auto TaskProto4TaskId = PlanUtil::MakeGetterTaskProto4TaskId(*plan);
  const auto& process_ranks = Singleton<ResourceDesc, ForSession>::Get()->process_ranks();
  FOR_RANGE(int32_t, i, 0, Singleton<CriticalSectionDesc>::Get()->CriticalSectionNum()) {
    const CriticalSection& cs = Singleton<CriticalSectionDesc>::Get()->GetCriticalSection(i);
    for (int64_t machine_id : process_ranks) {
      TaskProto* identity_tick =
          sole_tick_op_name2sole_task.at(identity_tick_op_names.at(i).at(machine_id));
      LinkTickTaskProto(
          plan, identity_tick,
          sole_tick_op_name2sole_task.at(cs.machine_id2source_tick_op_name().at(machine_id)),
          sole_tick_op_name2sole_task.at(cs.machine_id2sink_tick_op_name().at(machine_id)));
    }
  }
  {
    // erase source_tick task_proto
    HashSet<std::string> source_tick_op_names;
    FOR_RANGE(int32_t, i, 0, Singleton<CriticalSectionDesc>::Get()->CriticalSectionNum()) {
      const CriticalSection& cs = Singleton<CriticalSectionDesc>::Get()->GetCriticalSection(i);
      for (int64_t machine_id : process_ranks) {
        const auto& src_tick_op_name = cs.machine_id2source_tick_op_name().at(machine_id);
        CHECK(source_tick_op_names.emplace(src_tick_op_name).second);
      }
    }
    Erase<PbRpf<TaskProto>>(*plan->mutable_task(), [&](const TaskProto& task) {
      if (task.task_type() == TaskType::kSourceTick) {
        CHECK(task.exec_sequence().exec_node_size() == 1);
        const auto& kernel_conf = task.exec_sequence().exec_node(0).kernel_conf();
        const auto& op_conf = PlanUtil::GetOpAttribute(plan, task.job_id(), kernel_conf).op_conf();
        CHECK(op_conf.has_source_tick_conf());
        CHECK(source_tick_op_names.find(op_conf.name()) != source_tick_op_names.end());
        return true;
      } else {
        return false;
      }
    });
  }
}

void GetMemSharingOpBlobInfo(const JobBuilder& job_builder, const std::string& op_name,
                             ParallelBlobConf* blob_conf) {
  std::string obn = "out";
  std::string lbn;
  {
    const auto& op_conf = CHECK_JUST(job_builder.OpConf4OpName(op_name));
    if (op_conf.has_variable_conf()) {
      lbn = op_name + "/" + op_conf.variable_conf().out();
    } else if (op_conf.has_input_conf()) {
      lbn = op_name + "/" + op_conf.input_conf().out();
    } else if (op_conf.has_output_conf()) {
      lbn = op_name + "/" + op_conf.output_conf().out();
    } else if (op_conf.has_return_conf()) {
      lbn = op_name + "/" + op_conf.return_conf().out();
    } else {
      UNIMPLEMENTED();
    }
  }
  const auto& job = job_builder.job();
  ParallelBlobConf ret;
  *blob_conf->mutable_parallel_conf() = CHECK_JUST(job_builder.ParallelConf4OpName(op_name));
  *blob_conf->mutable_logical_blob_desc_conf() = job.helper().lbn2logical_blob_desc().at(lbn);
  *blob_conf->mutable_nd_sbp() =
      job.job_parallel_view_conf().op_name2nd_sbp_signature_conf().at(op_name).bn_in_op2nd_sbp().at(
          obn);
}

void FilterOpName2ParallelBlobConf(
    const HashSet<OperatorConf::OpTypeCase>& match, const std::vector<std::shared_ptr<Job>>& jobs,
    HashMap<std::string, ParallelBlobConf>* op_name2parallel_blob_conf) {
  FOR_RANGE(int64_t, job_id, 0, jobs.size()) {
    JobBuilder job_builder(jobs.at(job_id).get());
    for (const OperatorConf& op_conf : jobs.at(job_id)->net().op()) {
      if (match.find(op_conf.op_type_case()) == match.end()) { continue; }
      ParallelBlobConf parallel_blob_conf;
      GetMemSharingOpBlobInfo(job_builder, op_conf.name(), &parallel_blob_conf);
      auto iter = op_name2parallel_blob_conf->find(op_conf.name());
      if (iter == op_name2parallel_blob_conf->end()) {
        CHECK(op_name2parallel_blob_conf->emplace(op_conf.name(), parallel_blob_conf).second);
      } else {
        CHECK(parallel_blob_conf == iter->second);
      }
    }
  }
}

void CheckNonDistributeOptimizerAvailable(const std::vector<std::shared_ptr<Job>>& jobs) {
  bool has_job_enable_optimizer_placement_optimization = false;
  const auto IsEnabled = [](const Job& job) {
    return job.job_conf().has_train_conf()
           && job.job_conf().has_optimizer_placement_optimization_mode();
  };
  FOR_RANGE(int64_t, job_id, 0, jobs.size()) {
    if (IsEnabled(*jobs.at(job_id))) {
      has_job_enable_optimizer_placement_optimization = true;
      break;
    }
  }
  if (!has_job_enable_optimizer_placement_optimization) { return; }

  HashSet<std::string> var_names;
  FOR_RANGE(int64_t, job_id, 0, jobs.size()) {
    if (!IsEnabled(*jobs.at(job_id))) { continue; }
    for (const OperatorConf& op_conf : jobs.at(job_id)->net().op()) {
      if (op_conf.op_type_case() != OperatorConf::kVariableConf) { continue; }
      if (var_names.find(op_conf.name()) == var_names.end()) {
        var_names.emplace(op_conf.name());
      } else {
        // optimizer_placement_optimization jobs has a same variable in between them.
        LOG(FATAL)
            << "Only support optimizer_placement_optimization when jobs not sharing same variable";
      }
    }
  }
  FOR_RANGE(int64_t, job_id, 0, jobs.size()) {
    if (IsEnabled(*jobs.at(job_id))) { continue; }
    for (const OperatorConf& op_conf : jobs.at(job_id)->net().op()) {
      if (op_conf.op_type_case() != OperatorConf::kVariableConf) { continue; }
      if (var_names.find(op_conf.name()) != var_names.end()) {
        // Other jobs has a same variable in optimizer_placement_optimization jobs.
        LOG(FATAL)
            << "Only support optimizer_placement_optimization when jobs not sharing same variable";
      }
    }
  }
}

Maybe<ReentrantLockBackEdge> MakeMainJobComponent(
    const std::string& wait_and_send_ids_lbn, const Range& machine_id_range,
    JobBuilder* job_builder, std::vector<std::map<int64_t, std::string>>* identity_tick_op_names,
    std::vector<std::map<int64_t, std::string>>* cb_sink_tick_op_names) {
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name(std::string("@") + std::to_string(machine_id_range.begin()) + ":0");
  auto lock_back_edge = std::make_shared<ReentrantLockBackEdge>();
  OperatorConf reentrant_lock_op_conf;
  {
    lock_back_edge->reentrant_lock_op_name =
        std::string("System-Main-ReentrantLock_") + NewUniqueId();
    reentrant_lock_op_conf.set_name(lock_back_edge->reentrant_lock_op_name);
    auto* reentrant_lock_conf = reentrant_lock_op_conf.mutable_reentrant_lock_conf();
    reentrant_lock_conf->set_start(wait_and_send_ids_lbn);
    // ibn "end" is set after plan generated because we don't like cycle in job
    reentrant_lock_conf->set_out("out");
    Singleton<CriticalSectionDesc>::Get()->DumpCriticalSectionId2IntersectinIds(
        reentrant_lock_conf->mutable_lock_id2intersecting_lock_ids());
    JUST(job_builder->AddOp(parallel_conf, reentrant_lock_op_conf));
  }
  // critical section case op conf
  OperatorConf cs_case_op_conf;
  {
    cs_case_op_conf.set_name(std::string("System-Main-Case_") + NewUniqueId());
    auto* cs_case_conf = cs_case_op_conf.mutable_case_conf();
    cs_case_conf->set_in(reentrant_lock_op_conf.name() + "/out");
    FOR_RANGE(int64_t, i, 0, Singleton<CriticalSectionDesc>::Get()->CriticalSectionNum()) {
      cs_case_conf->add_out(GenRepeatedBn("out", i));
    }
    JUST(job_builder->AddOp(parallel_conf, cs_case_op_conf));
  }
  const int64_t num_critial_sections = Singleton<CriticalSectionDesc>::Get()->CriticalSectionNum();
  std::vector<std::string> snk_tick_op_names;
  snk_tick_op_names.reserve(num_critial_sections * machine_id_range.size());
  FOR_RANGE(int64_t, i, 0, num_critial_sections) {
    // source tick
    OperatorConf src_tick_op_conf;
    {
      std::string name_prefix = "System-Main-SourceTick_CriticalSection_";
      src_tick_op_conf.set_name(name_prefix + std::to_string(i) + "_" + NewUniqueId());
      auto* src_tick_conf = src_tick_op_conf.mutable_tick_conf();
      src_tick_conf->add_tick(cs_case_op_conf.name() + "/" + GenRepeatedBn("out", i));
      src_tick_conf->set_out("out");
      JUST(job_builder->AddOp(parallel_conf, src_tick_op_conf));
    }

    auto* cur_cb_sink_tick_op_names = &cb_sink_tick_op_names->at(i);
    for (int64_t machine_id = machine_id_range.begin(); machine_id < machine_id_range.end();
         ++machine_id) {
      // identity tick
      OperatorConf identity_tick_op_conf;
      {
        std::string name_prefix = "System-Main-Tick_CriticalSection_";
        identity_tick_op_conf.set_name(name_prefix + std::to_string(i) + "_" + NewUniqueId());
        auto* identity_tick_conf = identity_tick_op_conf.mutable_tick_conf();
        identity_tick_conf->add_tick(src_tick_op_conf.name() + "/out");
        identity_tick_conf->set_out("out");
        JUST(job_builder->AddOp(parallel_conf, identity_tick_op_conf));
        auto* cur_id_tick_op_names = &identity_tick_op_names->at(i);
        CHECK_OR_RETURN(
            cur_id_tick_op_names->emplace(machine_id, identity_tick_op_conf.name()).second);
      }
      // callback
      {
        OperatorConf cb_sink_tick_op_conf;
        std::string name_prefix = "System-Main-CallbackSinkTick_";
        cb_sink_tick_op_conf.set_name(name_prefix + std::to_string(i) + NewUniqueId());
        auto* cb_sink_tick_conf = cb_sink_tick_op_conf.mutable_sink_tick_conf();
        cb_sink_tick_conf->add_tick(identity_tick_op_conf.name() + "/out");
        cb_sink_tick_conf->set_out("out");
        JUST(job_builder->AddOp(parallel_conf, cb_sink_tick_op_conf));
        CHECK_OR_RETURN(
            cur_cb_sink_tick_op_names->emplace(machine_id, cb_sink_tick_op_conf.name()).second);
      }
      // sink tick
      {
        OperatorConf snk_tick_op_conf;
        std::string name_prefix = "System-Main-SinkTick_CriticalSection_";
        snk_tick_op_conf.set_name(name_prefix + std::to_string(i) + NewUniqueId());
        auto* snk_tick_conf = snk_tick_op_conf.mutable_sink_tick_conf();
        snk_tick_conf->add_tick(identity_tick_op_conf.name() + "/out");
        snk_tick_conf->set_out("out");
        JUST(job_builder->AddOp(parallel_conf, snk_tick_op_conf));
        snk_tick_op_names.emplace_back(snk_tick_op_conf.name());
      }
    }
  }
  // critical section esac op conf
  OperatorConf cs_esac_op_conf;
  {
    cs_esac_op_conf.set_name(std::string("System-Main-Esac_") + NewUniqueId());
    // cs_esac_op_conf.set_pass_tag("main");
    auto* cs_esac_conf = cs_esac_op_conf.mutable_esac_conf();
    for (const auto& snk_tick_op_name : snk_tick_op_names) {
      cs_esac_conf->add_in(snk_tick_op_name + "/out");
    }
    cs_esac_conf->set_out("out");
    cs_esac_conf->set_data_type(DataType::kInt32);
    JUST(job_builder->AddOp(parallel_conf, cs_esac_op_conf));
  }
  lock_back_edge->critical_section_sink_lbi.set_op_name(cs_esac_op_conf.name());
  lock_back_edge->critical_section_sink_lbi.set_blob_name("out");
  return lock_back_edge;
}

Maybe<void> MakeCallbackNotifierSinkTick(
    const std::set<int64_t>& process_ranks,
    const std::vector<std::map<int64_t, std::string>>& cb_sink_tick_op_names,
    JobBuilder* job_builder, const std::function<void(const std::string& lbn)>& DoEachSinkTickLbn) {
  const auto& MakeSinkTick = [&](const std::vector<int64_t>& job_cs_ids,
                                 int64_t machine_id) -> Maybe<std::string> {
    if (job_cs_ids.size() == 1) {
      return cb_sink_tick_op_names.at(job_cs_ids.at(0)).at(machine_id) + "/out";
    }
    ParallelConf machine_parallel_conf;
    {
      machine_parallel_conf.set_device_tag("cpu");
      machine_parallel_conf.add_device_name("@" + std::to_string(machine_id) + ":0");
    }
    OperatorConf snk_tick_op_conf;
    {
      std::string name_prefix = "System-Main-CallbackNotifier_CriticalSection_";
      snk_tick_op_conf.set_name(name_prefix + NewUniqueId());
      auto* snk_tick_conf = snk_tick_op_conf.mutable_sink_tick_conf();
      for (int64_t job_cs_id : job_cs_ids) {
        const auto& cb_sink_tick_op_name = cb_sink_tick_op_names.at(job_cs_id).at(machine_id);
        snk_tick_conf->add_tick(cb_sink_tick_op_name + "/out");
      }
      snk_tick_conf->set_out("out");
      JUST(job_builder->AddOp(machine_parallel_conf, snk_tick_op_conf));
    }
    return snk_tick_op_conf.name() + "/out";
  };
  ParallelConf parallel_conf;
  {
    parallel_conf.set_device_tag("cpu");
    parallel_conf.add_device_name("0:0");
  }
  for (const auto& cs_ids : Singleton<CriticalSectionDesc>::Get()->job_id2critical_section_ids()) {
    OperatorConf snk_tick_op_conf;
    {
      std::string name_prefix = "System-Main-CallbackNotifier_CriticalSection_";
      snk_tick_op_conf.set_name(name_prefix + NewUniqueId());
      snk_tick_op_conf.set_pass_tag(kMainOp);
      auto* snk_tick_conf = snk_tick_op_conf.mutable_sink_tick_conf();
      for (int64_t machine_id : process_ranks) {
        snk_tick_conf->add_tick(*JUST(MakeSinkTick(cs_ids, machine_id)));
      }
      snk_tick_conf->set_out("out");
      JUST(job_builder->AddOp(parallel_conf, snk_tick_op_conf));
    }
    DoEachSinkTickLbn(snk_tick_op_conf.name() + "/out");
  }
  return Maybe<void>::Ok();
}

Maybe<void> MakeMainJob(Job* main_job,
                        std::vector<std::map<int64_t, std::string>>* identity_tick_op_names,
                        std::vector<ReentrantLockBackEdge>* lock_back_edges) {
  JobBuilder job_builder(main_job);
  CHECK_OR_RETURN(GlobalProcessCtx::IsThisProcessMaster());
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0");
  OperatorConf wait_and_send_ids_op_conf;
  {
    wait_and_send_ids_op_conf.set_name(std::string("System-Main-WaitAndSendIds_") + NewUniqueId());
    wait_and_send_ids_op_conf.set_pass_tag(kMainOp);
    auto* wait_and_send_ids_conf = wait_and_send_ids_op_conf.mutable_wait_and_send_ids_conf();
    wait_and_send_ids_conf->set_out("out");
    wait_and_send_ids_conf->set_wait_buffer_name(kBufferNameGlobalWaitJobId);
    wait_and_send_ids_conf->set_data_type(DataType::kInt32);
    auto* id_list = wait_and_send_ids_conf->mutable_id_list();
    FOR_RANGE(int32_t, i, 0, Singleton<JobName2JobId>::Get()->size()) { id_list->Add(); }
    HashSet<int64_t> unique_check;
    for (const auto& pair : *Singleton<JobName2JobId>::Get()) {
      int64_t job_id = pair.second;
      CHECK_OR_RETURN(unique_check.insert(job_id).second);
      const auto& cs_idx = Singleton<CriticalSectionDesc>::Get()->CriticalSectionIds4JobId(job_id);
      *id_list->Mutable(job_id)->mutable_value() = {cs_idx.begin(), cs_idx.end()};
    }
    JUST(job_builder.AddOp(parallel_conf, wait_and_send_ids_op_conf));
  }
  const int64_t num_critial_sections = Singleton<CriticalSectionDesc>::Get()->CriticalSectionNum();
  std::vector<std::map<int64_t, std::string>> cb_sink_tick_op_names;
  identity_tick_op_names->resize(num_critial_sections);
  cb_sink_tick_op_names.resize(num_critial_sections);
  const auto& process_ranks = Singleton<ResourceDesc, ForSession>::Get()->process_ranks();
  for (int64_t machine_id : process_ranks) {
    Range sub_range(machine_id, machine_id + 1);
    const auto& in_lbn = wait_and_send_ids_op_conf.name() + "/out";
    lock_back_edges->emplace_back(*JUST(MakeMainJobComponent(
        in_lbn, sub_range, &job_builder, identity_tick_op_names, &cb_sink_tick_op_names)));
  }
  OperatorConf callback_notify_esac_op_conf;
  {
    callback_notify_esac_op_conf.set_name(std::string("System-Main-Esac_") + NewUniqueId());
    callback_notify_esac_op_conf.set_pass_tag(kMainOp);
    auto* callback_notify_esac_conf = callback_notify_esac_op_conf.mutable_esac_conf();
    JUST(MakeCallbackNotifierSinkTick(
        process_ranks, cb_sink_tick_op_names, &job_builder,
        [&](const std::string& lbn) { callback_notify_esac_conf->add_in(lbn); }));
    callback_notify_esac_conf->set_out("out");
    callback_notify_esac_conf->set_data_type(DataType::kInt32);
    JUST(job_builder.AddOp(parallel_conf, callback_notify_esac_op_conf));
  }
  OperatorConf callback_notify_op_conf;
  {
    callback_notify_op_conf.set_name(std::string("System-Main-CallbackNotify_") + NewUniqueId());
    callback_notify_op_conf.set_pass_tag(kMainOp);
    auto* callback_notify_conf = callback_notify_op_conf.mutable_callback_notify_conf();
    callback_notify_conf->set_in(callback_notify_esac_op_conf.name() + "/out");
    auto* buffer_names = callback_notify_conf->mutable_callback_buffer_name();
    FOR_RANGE(int64_t, i, 0, Singleton<JobName2JobId>::Get()->size()) { buffer_names->Add(); }
    for (const auto& pair : *Singleton<JobName2JobId>::Get()) {
      int64_t job_id = pair.second;
      const auto& buffer_name = GetCallbackNotifierBufferName(pair.first);
      *buffer_names->Mutable(job_id) = buffer_name;
    }
    JUST(job_builder.AddOp(parallel_conf, callback_notify_op_conf));
  }

  auto* job_conf = main_job->mutable_job_conf();
  job_conf->set_job_name("MainJob-unamed");
  job_conf->mutable_predict_conf();
  job_conf->set_default_data_type(DataType::kInt32);
  return Maybe<void>::Ok();
}

Maybe<void> ConnectCriticalSectionEndToReentrantLockEnd(
    Plan* main_plan, const ReentrantLockBackEdge& lock_back_edge) {
  TaskProto* reentrant_lock_task = nullptr;
  TaskProto* cs_sink_task = nullptr;
  FOR_RANGE(int64_t, i, 0, main_plan->task_size()) {
    auto* task = main_plan->mutable_task(i);
    CHECK_EQ_OR_RETURN(task->exec_sequence().exec_node_size(), 1);
    const auto& kernel_conf = task->exec_sequence().exec_node(0).kernel_conf();
    const auto& op_name =
        PlanUtil::GetOpAttribute(main_plan, task->job_id(), kernel_conf).op_conf().name();
    if (op_name == lock_back_edge.reentrant_lock_op_name) {
      CHECK_ISNULL_OR_RETURN(reentrant_lock_task);
      reentrant_lock_task = task;
    } else if (op_name == lock_back_edge.critical_section_sink_lbi.op_name()) {
      CHECK_ISNULL_OR_RETURN(cs_sink_task);
      cs_sink_task = task;
    } else {
      // do nothing
    }
  }
  CHECK_NOTNULL_OR_RETURN(reentrant_lock_task);
  CHECK_NOTNULL_OR_RETURN(cs_sink_task);
  RegstDescProto* cs_end_regst = PlanUtil::GetSoleProducedDataRegst(cs_sink_task);
  cs_end_regst->add_consumer_task_id(reentrant_lock_task->task_id());
  reentrant_lock_task->mutable_consumed_regst_desc_id()->at("in").add_regst_desc_id(
      cs_end_regst->regst_desc_id());

  auto* reentrant_exec_node = reentrant_lock_task->mutable_exec_sequence()->mutable_exec_node(0);
  (*reentrant_exec_node->mutable_bn_in_op2regst_desc_id())["end"] = cs_end_regst->regst_desc_id();

  auto* op_attribute = reentrant_exec_node->mutable_kernel_conf()->mutable_op_attribute();
  op_attribute->add_input_bns("end");
  (*op_attribute->mutable_arg_signature()->mutable_bn_in_op2lbi())["end"] =
      lock_back_edge.critical_section_sink_lbi;
  const auto& blob_desc_signature_map =
      op_attribute->logical_blob_desc_signature().bn_in_op2blob_desc();
  const auto it = blob_desc_signature_map.find("start");
  CHECK_OR_RETURN(it != blob_desc_signature_map.end());
  CHECK_OR_RETURN(blob_desc_signature_map.find("end") == blob_desc_signature_map.end());
  (*op_attribute->mutable_logical_blob_desc_signature()->mutable_bn_in_op2blob_desc())["end"] =
      it->second;
  auto* reentrant_lock_conf = op_attribute->mutable_op_conf()->mutable_reentrant_lock_conf();
  reentrant_lock_conf->set_end(GenLogicalBlobName(lock_back_edge.critical_section_sink_lbi));
  return Maybe<void>::Ok();
}

Maybe<void> CompileMainJob(Job* main_job, const std::vector<ReentrantLockBackEdge>& lock_back_edges,
                           int64_t job_id, Plan* main_plan) {
  CHECK_OR_RETURN(GlobalProcessCtx::IsThisProcessMaster());
  {
    auto scope = std::make_unique<GlobalJobDescScope>(main_job->job_conf(), job_id);
    JUST(CompileCurJobOnMaster(main_job, main_plan, false));
  }
  for (const auto& lock_back_edge : lock_back_edges) {
    JUST(ConnectCriticalSectionEndToReentrantLockEnd(main_plan, lock_back_edge));
  }
  return Maybe<void>::Ok();
}

void AddJobName2JobId(const std::string& job_name, int64_t job_id) {
  if (!GlobalProcessCtx::IsThisProcessMaster()) { return; }
  CHECK(Singleton<JobName2JobId>::Get()->emplace(job_name, job_id).second);
}

bool NeedAllocateMemory(const RegstDescTypeProto& regst_desc_type) {
  return regst_desc_type.has_data_regst_desc();
}

void FinishGlobalCriticalSectionDesc(const Plan& plan, int64_t job_size) {
  std::vector<HashMap<std::string, HashSet<int64_t>>> job_id2sole_op_name2mem_block_ids(job_size);
  std::vector<HashSet<int64_t>> job_id2mem_block_ids(job_size);
  std::vector<HashSet<int64_t>> job_id2chunk_ids(job_size);
  for (const auto& task : plan.task()) {
    if (task.exec_sequence().exec_node_size() == 1) {
      const auto& kernel_conf = task.exec_sequence().exec_node(0).kernel_conf();
      const std::string& op_name =
          PlanUtil::GetOpAttribute(&plan, task.job_id(), kernel_conf).op_conf().name();
      HashSet<int64_t>* mem_block_ids =
          &(job_id2sole_op_name2mem_block_ids.at(task.job_id())[op_name]);
      for (const auto& pair : task.produced_regst_desc()) {
        if (NeedAllocateMemory(pair.second.regst_desc_type())) {
          mem_block_ids->emplace(pair.second.mem_block_id());
        }
        if (pair.second.has_separated_header_mem_block_id()
            && pair.second.separated_header_mem_block_id() != -1) {
          mem_block_ids->emplace(pair.second.separated_header_mem_block_id());
        }
      }
    }
  }
  for (const auto& mem_block : plan.block_chunk_list().mem_block()) {
    if (mem_block.mem_size() == 0) { continue; }
    for (int64_t job_id : mem_block.job_id()) {
      job_id2mem_block_ids.at(job_id).insert(mem_block.mem_block_id());
    }
  }
  for (const auto& chunk : plan.block_chunk_list().chunk()) {
    if (chunk.mem_size() == 0) { continue; }
    for (int64_t job_id : chunk.job_id()) { job_id2chunk_ids.at(job_id).insert(chunk.chunk_id()); }
  }

  HashMap<int64_t, HashSet<int64_t>> job_id2input_output_mem_block_ids;
  auto* critical_section_desc = Singleton<CriticalSectionDesc>::Get();
  // set mem_block_id for InputOutputCriticalSection
  FOR_RANGE(int64_t, i, 0, critical_section_desc->CriticalSectionNum()) {
    auto* critical_section = critical_section_desc->MutCriticalSection(i);
    int64_t job_id = critical_section->job_id();
    auto* input_output_mem_block_ids = &job_id2input_output_mem_block_ids[job_id];
    if (critical_section->has_input_output_critical_section()) {
      HashSet<int64_t> mem_block_ids;
      for (const auto& op_name :
           critical_section->input_output_critical_section().lbi_producer_op_name()) {
        const auto& cur_mem_block_ids = job_id2sole_op_name2mem_block_ids.at(job_id).at(op_name);
        mem_block_ids.insert(cur_mem_block_ids.begin(), cur_mem_block_ids.end());
      }
      *critical_section->mutable_mem_block_id() = {mem_block_ids.begin(), mem_block_ids.end()};
      input_output_mem_block_ids->insert(mem_block_ids.begin(), mem_block_ids.end());
    } else {
      CHECK(critical_section->has_total_job_critical_section());
    }
  }
  HashSet<int64_t> unique_job_id_check;
  // set mem_block_id for TotalJobCriticalSection
  FOR_RANGE(int64_t, i, 0, critical_section_desc->CriticalSectionNum()) {
    auto* critical_section = critical_section_desc->MutCriticalSection(i);
    int64_t job_id = critical_section->job_id();
    const auto& input_output_mem_block_ids = job_id2input_output_mem_block_ids.at(job_id);
    if (critical_section->has_total_job_critical_section()) {
      CHECK(unique_job_id_check.emplace(job_id).second);
      auto* mem_block_ids = &job_id2mem_block_ids.at(job_id);
      {
        // exclude input/output criticalsection mem_blob_ids from total_job
        auto it = mem_block_ids->begin();
        while (it != mem_block_ids->end()) {
          if (input_output_mem_block_ids.find(*it) == input_output_mem_block_ids.end()) {
            ++it;
          } else {
            it = mem_block_ids->erase(it);
          }
        }
      }
      *critical_section->mutable_mem_block_id() = {mem_block_ids->begin(), mem_block_ids->end()};
      *critical_section->mutable_chunk_id() = {job_id2chunk_ids.at(job_id).begin(),
                                               job_id2chunk_ids.at(job_id).end()};
    }
  }
  critical_section_desc->Done();
}

REGISTER_FUNCTION_CONFIG_DEF().Bool("__is_user_function__", true, "is user defined function");

Maybe<void> CompileJobsAndMergePlans(const PbRpf<Job>& job_confs, Plan& plan) {
  std::vector<std::shared_ptr<Job>> jobs(job_confs.size());
  FOR_RANGE(int, i, 0, jobs.size()) { jobs.at(i).reset(new Job(job_confs.Get(i))); }
  // These checks donot work in nn.Graph API because there is only on job compile each time.
  // And nn.Graph Support training and evaluation share the same variable.
  if (jobs.size() > 1) { CheckNonDistributeOptimizerAvailable(jobs); }
  HashMap<std::string, ParallelBlobConf> var_op_name2parallel_blob_conf;
  FilterOpName2ParallelBlobConf({OperatorConf::kVariableConf}, jobs,
                                &var_op_name2parallel_blob_conf);
  std::vector<std::shared_ptr<Job>> function_jobs;
  function_jobs.reserve(jobs.size());
  FOR_RANGE(int, i, 0, jobs.size()) {
    JobDesc job_desc(jobs.at(i)->job_conf(), i);
    if (job_desc.Bool("__is_user_function__")) { function_jobs.emplace_back(jobs.at(i)); }
  }

  std::vector<Plan> sub_plans(jobs.size());
  FOR_RANGE(int64_t, i, 0, jobs.size()) {
    AddJobName2JobId(jobs.at(i)->job_conf().job_name(), i);
    auto scope = std::make_unique<GlobalJobDescScope>(jobs.at(i)->job_conf(), i);
    JUST(CompileCurJobOnMaster(jobs.at(i).get(), &sub_plans.at(i), true));
  }
  MergeSubPlan(&plan, std::move(sub_plans));
  InterJobMemSharingUtil::MergeMemReusedChunkBetweenUserJobs(function_jobs, &plan);
  InterJobMemSharingUtil::MergeMemSharedInterfaceMemBlockBetweenJobs(jobs, &plan);
  PlanUtil::SetForceInplaceMemBlock(&plan);
  FinishGlobalCriticalSectionDesc(plan, jobs.size());
  Plan main_plan;
  std::vector<std::map<int64_t, std::string>> identity_tick_op_names;
  {
    Job main_job;
    std::vector<ReentrantLockBackEdge> lock_back_edges;
    JUST(MakeMainJob(&main_job, &identity_tick_op_names, &lock_back_edges));
    AddJobName2JobId(main_job.job_conf().job_name(), jobs.size());
    JUST(CompileMainJob(&main_job, lock_back_edges, jobs.size(), &main_plan));
  }
  LinkMainPlan(&plan, std::move(main_plan), identity_tick_op_names);
  PlanUtil::CleanUselessMemBlockAndCheckValid(&plan);
  PlanUtil::DumpCtrlRegstInfoToPlan(&plan);
  PlanUtil::PlanMemoryLog(&plan, "merged_plan");
  if (Singleton<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
    TeePersistentLogStream::Create("merged_plan")->Write(plan);
    PlanUtil::ToDotFile(plan, "/dot/merged_plan.dot");
  }
  return Maybe<void>::Ok();
}

Maybe<void> CompileJobsAndPushMergedPlan(const PbRpf<Job>& job_confs) {
  if (GlobalProcessCtx::IsThisProcessMaster()) {
    Plan plan;
    JUST(CompileJobsAndMergePlans(job_confs, plan));
    double start = GetCurTime();
    // push op_attribute_info
    OpAttributeInfo op_attribute_info;
    *op_attribute_info.mutable_job_id2op_attribute_ref_table() =
        plan.job_id2op_attribute_ref_table();
    Singleton<CtrlClient>::Get()->PushKV("op_attribute_info", op_attribute_info);
    // push plan
    PushPlan("merged_plan", std::move(plan));
    LOG(INFO) << " PushPlan merged_plan time: " << (GetCurTime() - start) / 1e9 << " seconds.\n";
  }
  OF_SESSION_BARRIER();
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> Oneflow::Init(const oneflow::JobSet& job_set) {
  OF_PROFILER_RANGE_GUARD("Oneflow::Init");
  // Runtime
  OF_PROFILER_RANGE_PUSH("CompileJobsAndPushMergedPlan");
  JUST(CompileJobsAndPushMergedPlan(job_set.job()));
  OF_PROFILER_RANGE_POP();  // CompileJobsAndPushMergedPlan
  double start = GetCurTime();
  PullPlan("merged_plan", &plan_);
  LOG(INFO) << " PullPlan merged_plan time: " << (GetCurTime() - start) / 1e9 << " seconds.\n";
  if (GlobalProcessCtx::IsThisProcessMaster()) {
    runtime_buffers_scope_.reset(new RuntimeBuffersScope(plan_.job_confs()));
  }
  OF_PROFILER_RANGE_PUSH("new Runtime");

  HashMap<std::string, vm::EagerBlobObject*> variable_op_name2eager_blob_object;
  runtime_.reset(new Runtime(plan_, variable_op_name2eager_blob_object));
  OF_PROFILER_RANGE_POP();  // new Runtime
  return Maybe<void>::Ok();
}

Oneflow::~Oneflow() {
  if (GlobalProcessCtx::IsThisProcessMaster()) { runtime_buffers_scope_.reset(); }
  runtime_.reset();
}

}  // namespace oneflow
