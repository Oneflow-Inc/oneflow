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
#include "oneflow/core/common/range.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/job/compiler.h"
#include "oneflow/core/job/improver.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/profiler.h"
#include "oneflow/core/job/sub_plan.pb.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/available_memory_desc.pb.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/actor/act_event_logger.h"
#include "oneflow/core/job/oneflow.h"
#include "oneflow/core/job/model_io_v2_job.h"
#include "oneflow/core/job/model_io_job.h"
#include "oneflow/core/job/inter_job_mem_sharing_util.h"
#include "oneflow/core/job/plan_util.h"
#include "oneflow/core/operator/interface_op_util.h"
#include "oneflow/core/job/critical_section_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/vm/oneflow_vm.h"
#include "oneflow/core/graph/plan_task_graph.h"
#include "oneflow/core/graph/boxing/collective_boxing_util.h"
#include "oneflow/core/profiler/profiler.h"

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
         && lhs.parallel_conf() == rhs.parallel_conf() && lhs.sbp_conf() == rhs.sbp_conf();
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

std::string net_topo_key(const std::string& plan_name) { return plan_name + "_net_topo"; }

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

std::shared_ptr<OperatorConf> CreateSinkTickOpConf(const std::string& in_op_name) {
  auto tick_op = std::make_shared<OperatorConf>();
  tick_op->set_name("System-Main-CallbackNotifier_TmpSinkTick_" + NewUniqueId());
  auto* tick_conf = tick_op->mutable_sink_tick_conf();
  tick_conf->add_tick(in_op_name + "/out");
  tick_conf->set_out("out");
  return tick_op;
}

void PushPlan(const std::string& plan_name, const Plan& plan) {
  HashMap<int64_t, std::set<int64_t>> machine_id2thrd_id_set;
  HashMap<std::pair<int64_t, int64_t>, std::vector<TaskProto>> mchn_thrd_id2task_protos;
  HashMap<int64_t, MemBlockAndChunkList> machine_id2block7chunk;

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

  for (const auto& mem_block : plan.block_chunk_list().mem_block()) {
    *machine_id2block7chunk[mem_block.machine_id()].add_mem_block() = mem_block;
  }
  for (const auto& chunk : plan.block_chunk_list().chunk()) {
    *machine_id2block7chunk[chunk.machine_id()].add_chunk() = chunk;
  }
  for (const auto& pair : machine_id2block7chunk) {
    Global<CtrlClient>::Get()->PushKV(block7chunk_key(plan_name, pair.first), pair.second);
  }

  Global<CtrlClient>::Get()->PushKV(net_topo_key(plan_name), plan.net_topo());
  Global<CtrlClient>::Get()->PushKV(job_id2job_conf(plan_name), plan.job_confs());
  Global<CtrlClient>::Get()->PushKV(GetCollectiveBoxingPlanKey(plan_name),
                                    plan.collective_boxing_plan());
}

void PullPlan(const std::string& plan_name, Plan* plan) {
  ClusterThrdIds cluster_thrd_ids;
  Global<CtrlClient>::Get()->PullKV(cluster_thrd_ids_key(plan_name), &cluster_thrd_ids);
  PrintProtoToTextFile(cluster_thrd_ids, JoinPath(FLAGS_log_dir, cluster_thrd_ids_key(plan_name)));
  HashMap<int64_t, ThrdIds> machine_id2thrd_ids;
  machine_id2thrd_ids = PbMap2HashMap(cluster_thrd_ids.machine_id2thrd_ids());
  int64_t machine_id = GlobalProcessCtx::Rank();
  auto thrd_ids_it = machine_id2thrd_ids.find(machine_id);
  CHECK(thrd_ids_it != machine_id2thrd_ids.end());
  std::vector<int64_t> thrd_id_vec = PbRf2StdVec(thrd_ids_it->second.thrd_id());
  for (auto thrd_id : thrd_id_vec) {
    SubPlan sub_plan;
    Global<CtrlClient>::Get()->PullKV(sub_plan_key(plan_name, machine_id, thrd_id), &sub_plan);
    plan->mutable_task()->MergeFrom(sub_plan.task());
  }
  NetTopo net_topo;
  Global<CtrlClient>::Get()->PullKV(net_topo_key(plan_name), &net_topo);
  *(plan->mutable_net_topo()) = net_topo;
  JobConfs job_confs;
  Global<CtrlClient>::Get()->PullKV(job_id2job_conf(plan_name), &job_confs);
  *(plan->mutable_job_confs()) = job_confs;
  Global<CtrlClient>::Get()->PullKV(GetCollectiveBoxingPlanKey(plan_name),
                                    plan->mutable_collective_boxing_plan());
  MemBlockAndChunkList block7chunk;
  Global<CtrlClient>::Get()->PullKV(block7chunk_key(plan_name, machine_id), &block7chunk);
  plan->mutable_block_chunk_list()->CopyFrom(block7chunk);
}

bool IsCollectiveBoxingNode(const PlanTaskNode* node) {
  const TaskType task_type = node->task_proto()->task_type();
  return task_type == TaskType::kCollectiveBoxingGeneric;
}

const boxing::collective::RankDesc& GetRankDesc(const OperatorConf& conf) {
  if (conf.has_collective_boxing_generic_conf()) {
    return conf.collective_boxing_generic_conf().rank_desc();
  } else {
    UNIMPLEMENTED();
  }
}

const boxing::collective::RankDesc& GetRankDesc(const TaskProto& task_proto) {
  CHECK_EQ(task_proto.exec_sequence().exec_node_size(), 1);
  return GetRankDesc(
      task_proto.exec_sequence().exec_node(0).kernel_conf().op_attribute().op_conf());
}

void GetDeviceDesc(const TaskProto* task_proto, boxing::collective::DeviceDesc* device_desc) {
  device_desc->set_machine_id(task_proto->machine_id());
  const int64_t thrd_id = Global<IDMgr>::Get()->ThrdId4ActorId(task_proto->task_id());
  device_desc->set_device_type(Global<IDMgr>::Get()->GetDeviceTypeFromThrdId(thrd_id));
  if (device_desc->device_type() == DeviceType::kGPU) {
    device_desc->set_device_id(Global<IDMgr>::Get()->GetGpuPhyIdFromThrdId(thrd_id));
  } else {
    UNIMPLEMENTED();
  }
}

void GenCollectiveBoxingPlan(Job* job, Plan* plan) {
  using namespace boxing::collective;

  struct RequestInfo {
    OpDesc op_desc;
    std::map<int64_t, const PlanTaskNode*> rank2node;
    int64_t order;
    int64_t dependency_depth;
  };

  PlanTaskGraph plan_task_graph(*plan);
  int64_t dependency_depth = 0;
  int64_t order = 0;
  RequestSet* request_set = &(*plan->mutable_collective_boxing_plan()
                                   ->mutable_job_id2request_set())[GlobalJobDesc().job_id()];
  HashSet<const PlanTaskNode*> all_visited;
  while (true) {
    std::list<const PlanTaskNode*> src_nodes;
    plan_task_graph.ForEachNode([&](const PlanTaskNode* node) {
      if (all_visited.count(node) != 0) { return; }
      int64_t in_cnt = 0;
      node->ForEachNodeOnInEdge([&](const PlanTaskNode* node_on_in_edge) {
        if (all_visited.count(node_on_in_edge) != 0) { return; }
        in_cnt += 1;
      });
      if (in_cnt == 0) { src_nodes.push_back(node); }
    });
    if (src_nodes.empty()) { break; }
    auto ForEachNodeOnInEdge = [&](const PlanTaskNode* node,
                                   const std::function<void(const PlanTaskNode*)>& Handler) {
      node->ForEachNodeOnInEdge([&](const PlanTaskNode* node_on_in_edge) {
        if (all_visited.count(node_on_in_edge) == 0) { Handler(node_on_in_edge); }
      });
    };
    auto ForEachNodeOnOutEdge = [&](const PlanTaskNode* node,
                                    const std::function<void(const PlanTaskNode*)>& Handler) {
      if (!IsCollectiveBoxingNode(node)) {
        node->ForEachNodeOnOutEdge([&](const PlanTaskNode* node_on_out_edge) {
          bool has_unvisited_collective_boxing_node_on_in_edges = false;
          node_on_out_edge->ForEachNodeOnInEdge([&](const PlanTaskNode* node_on_in_edge) {
            if (!has_unvisited_collective_boxing_node_on_in_edges
                && IsCollectiveBoxingNode(node_on_in_edge)
                && all_visited.count(node_on_in_edge) == 0) {
              has_unvisited_collective_boxing_node_on_in_edges = true;
            }
          });
          if (!has_unvisited_collective_boxing_node_on_in_edges) { Handler(node_on_out_edge); }
        });
      }
    };
    HashSet<const PlanTaskNode*> visited;
    std::vector<const PlanTaskNode*> collective_boxing_nodes;
    plan_task_graph.TopoForEachNode(src_nodes, ForEachNodeOnInEdge, ForEachNodeOnOutEdge,
                                    [&](const PlanTaskNode* node) {
                                      visited.insert(node);
                                      if (IsCollectiveBoxingNode(node)) {
                                        collective_boxing_nodes.push_back(node);
                                      }
                                    });
    if (collective_boxing_nodes.empty()) { break; }
    HashMap<std::string, RequestInfo> name2request_info;
    for (const PlanTaskNode* node : collective_boxing_nodes) {
      const TaskProto* task_proto = node->task_proto();
      const RankDesc& rank_desc = GetRankDesc(*task_proto);
      CHECK_GE(rank_desc.rank(), 0);
      CHECK_LT(rank_desc.rank(), rank_desc.op_desc().num_ranks());
      const std::string& name = rank_desc.op_desc().name();
      boxing::collective::DeviceDesc device_desc;
      GetDeviceDesc(task_proto, &device_desc);
      auto it = name2request_info.find(name);
      if (it == name2request_info.end()) {
        RequestInfo request_info{
            .op_desc = rank_desc.op_desc(),
            .rank2node = {std::make_pair(rank_desc.rank(), node)},
            .order = order,
            .dependency_depth = dependency_depth,
        };
        name2request_info.emplace(std::make_pair(name, std::move(request_info)));
        order += 1;
      } else {
        CHECK(it->second.op_desc == rank_desc.op_desc());
        CHECK(it->second.rank2node.emplace(std::make_pair(rank_desc.rank(), node)).second);
      }
    }
    int64_t collected = 0;
    for (const auto& name7request_info : name2request_info) {
      const RequestInfo& info = name7request_info.second;
      if (info.rank2node.size() == info.op_desc.num_ranks()) {
        collected += 1;
        boxing::collective::RequestDesc* request_desc = request_set->mutable_request()->Add();
        *request_desc->mutable_op_desc() = info.op_desc;
        for (int64_t i = 0; i < info.op_desc.num_ranks(); ++i) {
          GetDeviceDesc(info.rank2node.at(i)->task_proto(),
                        request_desc->mutable_device_set()->mutable_device()->Add());
        }
        request_desc->set_order(info.order);
        request_desc->set_dependency_depth(info.dependency_depth);
      } else {
        CHECK_LT(info.rank2node.size(), info.op_desc.num_ranks());
        for (const auto& pair : info.rank2node) { visited.erase(pair.second); }
      }
    }
    CHECK_GT(collected, 0);
    all_visited.insert(visited.begin(), visited.end());
    ++dependency_depth;
  }
}

Maybe<void> CompileCurJobOnMaster(Job* job, Plan* improved_plan, bool need_job_complete) {
  const JobDesc& job_desc = GlobalJobDesc();
  Plan naive_plan;
  Plan complete_plan;
  double start = GetCurTime();
  if (GlobalProcessCtx::IsThisProcessMaster()) {
    Compiler().Compile(job, &naive_plan, need_job_complete);
    LOG(INFO) << "compile time: " << GetCurTime() - start;
    complete_plan =
        *JUST(Improver().GenAndInferMemBlockIdOnly(*Global<AvailableMemDesc>::Get(), naive_plan));
    if (Global<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
      TeePersistentLogStream::Create("naive_plan")->Write(naive_plan);
      TeePersistentLogStream::Create("complete_plan")->Write(complete_plan);
    }
    LOG(INFO) << "push_pull_plan:" << GetCurTime() - start;
  }
  if (job_desc.enable_experiment_run()) {
    if (GlobalProcessCtx::IsThisProcessMaster()) {
      PushPlan("complete_plan", complete_plan);
    } else {
      PullPlan("complete_plan", &complete_plan);
    }
    OF_SESSION_BARRIER();
    // Experiment Runtime
    { Runtime experiment_run(complete_plan, job_desc.piece_num_of_experiment_phase(), true); }
    // Improve
    if (GlobalProcessCtx::IsThisProcessMaster()) {
      TeePersistentLogStream::Create("available_mem_desc")->Write(*Global<AvailableMemDesc>::Get());
      CHECK_GT(Global<AvailableMemDesc>::Get()->machine_amd_size(), 0);
      *improved_plan = *JUST(Improver().Improve(
          *Global<AvailableMemDesc>::Get(), naive_plan,
          JoinPath(FLAGS_log_dir, ActEventLogger::experiment_act_event_bin_filename())));
      OF_SESSION_BARRIER();
      TeePersistentLogStream::Create("improved_plan")->Write(*improved_plan);
    }
  } else {
    *improved_plan = complete_plan;
  }
  GenCollectiveBoxingPlan(job, improved_plan);
  LOG(INFO) << "compile and improve time: " << GetCurTime() - start;
  return Maybe<void>::Ok();
}

void MergePlanWithoutGenNetTopo(Plan* plan, const Plan& other) {
  plan->mutable_task()->MergeFrom(other.task());
  plan->mutable_block_chunk_list()->MergeFrom(other.block_chunk_list());

  for (const auto& pair : other.job_confs().job_id2job_conf()) {
    CHECK(plan->mutable_job_confs()->mutable_job_id2job_conf()->insert(pair).second);
  }
  for (const auto& pair : other.collective_boxing_plan().job_id2request_set()) {
    CHECK(
        plan->mutable_collective_boxing_plan()->mutable_job_id2request_set()->insert(pair).second);
  }
}

void MergeSubPlanWithoutGenNetTopo(Plan* plan, const std::vector<Plan>& sub_plans) {
  CHECK(!sub_plans.empty());
  *plan = sub_plans.at(0);
  FOR_RANGE(int32_t, i, 1, sub_plans.size()) { MergePlanWithoutGenNetTopo(plan, sub_plans.at(i)); }
}

void MergePlan(Plan* plan, const Plan& other) {
  MergePlanWithoutGenNetTopo(plan, other);
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
//                        op_src_tick -->/
//
// note: after this function called, op_src_tick is illegal and need to be deleted from plan
void LinkTickTaskProto(TaskProto* identity_tick, TaskProto* src_tick, TaskProto* sink_tick) {
  CHECK(GetSoleOpConf(*identity_tick).has_tick_conf());
  CHECK(GetSoleOpConf(*src_tick).has_source_tick_conf());
  CHECK(GetSoleOpConf(*sink_tick).has_sink_tick_conf());
  RegstDescProto* id_tick_sole_regst = GetSoleDataRegstDescProto(identity_tick);
  RegstDescProto* src_tick_sole_regst = GetSoleDataRegstDescProto(src_tick);
  RegstDescProto* sink_tick_sole_regst = GetSoleDataRegstDescProto(sink_tick);

  sink_tick_sole_regst->set_regst_desc_id(id_tick_sole_regst->regst_desc_id());
  *sink_tick_sole_regst->mutable_consumer_task_id() = id_tick_sole_regst->consumer_task_id();
  UpdateSoleObnRegstDescId(sink_tick);
  CHECK_EQ(identity_tick->machine_id(), sink_tick->machine_id());

  id_tick_sole_regst->set_regst_desc_id(src_tick_sole_regst->regst_desc_id());
  *id_tick_sole_regst->mutable_consumer_task_id() = src_tick_sole_regst->consumer_task_id();
  UpdateSoleObnRegstDescId(identity_tick);
}

void FixRegstHostMemCase(TaskProto* task_proto,
                         const std::function<const TaskProto*(int64_t)>& TaskProto4TaskId) {
  for (auto& pair : *task_proto->mutable_produced_regst_desc()) {
    auto* regst = &pair.second;
    CHECK(regst->mem_case().has_host_mem());
    CHECK_EQ(regst->mem_case().host_mem().has_cuda_pinned_mem(), false);
    bool used_by_network = false;
    for (int64_t consumer_task_id : regst->consumer_task_id()) {
      const auto* consumer_task_proto = TaskProto4TaskId(consumer_task_id);
      used_by_network =
          used_by_network || (task_proto->machine_id() != consumer_task_proto->machine_id());
    }
    regst->mutable_mem_case()->mutable_host_mem()->set_used_by_network(used_by_network);
  }
}

void LinkMainPlan(Plan* plan, const Plan& main_plan,
                  const std::vector<std::map<int64_t, std::string>>& identity_tick_op_names) {
  std::function<bool(const TaskProto*)> IsInterfaceTickTockTask;
  {
    auto task_ids = std::make_shared<HashSet<int64_t>>();
    for (const auto& task : main_plan.task()) {
      if (task.task_type() == TaskType::kTick) { CHECK(task_ids->emplace(task.task_id()).second); }
    }
    IsInterfaceTickTockTask = [task_ids](const TaskProto* task) {
      if (task_ids->find(task->task_id()) != task_ids->end()) { return true; }
      if (task->exec_sequence().exec_node_size() != 1) { return false; }
      const auto& kernel_conf = task->exec_sequence().exec_node(0).kernel_conf();
      OperatorConf::OpTypeCase op_type_case = kernel_conf.op_attribute().op_conf().op_type_case();
      return op_type_case == OperatorConf::kSourceTickConf
             || op_type_case == OperatorConf::kSinkTickConf;
    };
  }
  MergePlan(plan, main_plan);
  HashMap<std::string, TaskProto*> sole_tick_op_name2sole_task;
  FOR_RANGE(int64_t, i, 0, plan->task_size()) {
    TaskProto* task = plan->mutable_task(i);
    if (IsInterfaceTickTockTask(task) == false) { continue; }
    const auto& kernel_conf = task->exec_sequence().exec_node(0).kernel_conf();
    const auto& op_name = kernel_conf.op_attribute().op_conf().name();
    CHECK(sole_tick_op_name2sole_task.emplace(op_name, task).second);
  }
  auto TaskProto4TaskId = PlanUtil::MakeGetterTaskProto4TaskId(*plan);
  int64_t num_machines = Global<ResourceDesc, ForSession>::Get()->TotalMachineNum();
  FOR_RANGE(int32_t, i, 0, Global<CriticalSectionDesc>::Get()->CriticalSectionNum()) {
    const CriticalSection& cs = Global<CriticalSectionDesc>::Get()->GetCriticalSection(i);
    for (int64_t machine_id = 0; machine_id < num_machines; ++machine_id) {
      TaskProto* identity_tick =
          sole_tick_op_name2sole_task.at(identity_tick_op_names.at(i).at(machine_id));
      LinkTickTaskProto(
          identity_tick,
          sole_tick_op_name2sole_task.at(cs.machine_id2source_tick_op_name().at(machine_id)),
          sole_tick_op_name2sole_task.at(cs.machine_id2sink_tick_op_name().at(machine_id)));
      FixRegstHostMemCase(identity_tick, TaskProto4TaskId);
    }
  }
  {
    // erase source_tick task_proto
    HashSet<std::string> source_tick_op_names;
    FOR_RANGE(int32_t, i, 0, Global<CriticalSectionDesc>::Get()->CriticalSectionNum()) {
      const CriticalSection& cs = Global<CriticalSectionDesc>::Get()->GetCriticalSection(i);
      for (int64_t machine_id = 0; machine_id < num_machines; ++machine_id) {
        const auto& src_tick_op_name = cs.machine_id2source_tick_op_name().at(machine_id);
        CHECK(source_tick_op_names.emplace(src_tick_op_name).second);
      }
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

void GetMemSharingOpBlobInfo(const JobBuilder& job_builder, const std::string& op_name,
                             ParallelBlobConf* blob_conf) {
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
    } else if (op_conf.has_return_conf()) {
      lbn = op_name + "/" + op_conf.return_conf().out();
    } else {
      UNIMPLEMENTED();
    }
  }
  const auto& job = job_builder.job();
  ParallelBlobConf ret;
  *blob_conf->mutable_parallel_conf() = job_builder.ParallelConf4OpName(op_name);
  *blob_conf->mutable_logical_blob_desc_conf() = job.helper().lbn2logical_blob_desc().at(lbn);
  *blob_conf->mutable_sbp_conf() = job.job_parallel_view_conf()
                                       .op_name2sbp_signature_conf()
                                       .at(op_name)
                                       .bn_in_op2sbp_parallel()
                                       .at(obn);
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
      if (op_conf.op_type_case() == OperatorConf::kVariableConf) { continue; }
      if (var_names.find(op_conf.name()) == var_names.end()) {
        var_names.emplace(op_conf.name());
      } else {
        LOG(FATAL)
            << "Only support optimizer_placement_optimization when jobs not sharing same variable";
      }
    }
  }
  FOR_RANGE(int64_t, job_id, 0, jobs.size()) {
    if (IsEnabled(*jobs.at(job_id))) { continue; }
    for (const OperatorConf& op_conf : jobs.at(job_id)->net().op()) {
      if (op_conf.op_type_case() == OperatorConf::kVariableConf) { continue; }
      if (var_names.find(op_conf.name()) != var_names.end()) {
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
  parallel_conf.add_device_name(std::to_string(machine_id_range.begin()) + ":0");
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
    Global<CriticalSectionDesc>::Get()->DumpCriticalSectionId2IntersectinIds(
        reentrant_lock_conf->mutable_lock_id2intersecting_lock_ids());
    JUST(job_builder->AddOp(parallel_conf, reentrant_lock_op_conf));
  }
  // critical section case op conf
  OperatorConf cs_case_op_conf;
  {
    cs_case_op_conf.set_name(std::string("System-Main-Case_") + NewUniqueId());
    auto* cs_case_conf = cs_case_op_conf.mutable_case_conf();
    cs_case_conf->set_in(reentrant_lock_op_conf.name() + "/out");
    FOR_RANGE(int64_t, i, 0, Global<CriticalSectionDesc>::Get()->CriticalSectionNum()) {
      cs_case_conf->add_out(GenRepeatedBn("out", i));
    }
    JUST(job_builder->AddOp(parallel_conf, cs_case_op_conf));
  }
  const int64_t num_critial_sections = Global<CriticalSectionDesc>::Get()->CriticalSectionNum();
  std::vector<std::string> snk_tick_op_names;
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
    // identity tick
    auto* cur_cb_sink_tick_op_names = &cb_sink_tick_op_names->at(i);
    for (int64_t machine_id = machine_id_range.begin(); machine_id < machine_id_range.end();
         ++machine_id) {
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
    }
    // sink tick
    {
      OperatorConf snk_tick_op_conf;
      std::string name_prefix = "System-Main-SinkTick_CriticalSection_";
      snk_tick_op_conf.set_name(name_prefix + std::to_string(i) + NewUniqueId());
      auto* snk_tick_conf = snk_tick_op_conf.mutable_sink_tick_conf();
      for (const auto& pair : *cur_cb_sink_tick_op_names) {
        snk_tick_conf->add_tick(pair.second + "/out");
      }
      snk_tick_conf->set_out("out");
      JUST(job_builder->AddOp(parallel_conf, snk_tick_op_conf));
      snk_tick_op_names.push_back(snk_tick_op_conf.name());
    }
  }
  // critical section esac op conf
  OperatorConf cs_esac_op_conf;
  {
    cs_esac_op_conf.set_name(std::string("System-Main-Esac_") + NewUniqueId());
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
    const Range& machine_id_range,
    const std::vector<std::map<int64_t, std::string>>& cb_sink_tick_op_names,
    JobBuilder* job_builder, const std::function<void(const std::string& lbn)>& DoEachSinkTickLbn) {
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0");
  for (int64_t total_job_cs_id :
       Global<CriticalSectionDesc>::Get()->job_id2total_job_critical_section_id()) {
    OperatorConf snk_tick_op_conf;
    {
      std::string name_prefix = "System-Main-CallbackNotifier_CriticalSection_";
      snk_tick_op_conf.set_name(name_prefix + std::to_string(total_job_cs_id));
      auto* snk_tick_conf = snk_tick_op_conf.mutable_sink_tick_conf();
      for (int64_t machine_id = machine_id_range.begin(); machine_id < machine_id_range.end();
           ++machine_id) {
        const auto& cb_sink_tick_op_name = cb_sink_tick_op_names.at(total_job_cs_id).at(machine_id);
        snk_tick_conf->add_tick(cb_sink_tick_op_name + "/out");
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
    auto* wait_and_send_ids_conf = wait_and_send_ids_op_conf.mutable_wait_and_send_ids_conf();
    wait_and_send_ids_conf->set_out("out");
    wait_and_send_ids_conf->set_wait_buffer_name(kBufferNameGlobalWaitJobId);
    wait_and_send_ids_conf->set_data_type(DataType::kInt32);
    auto* id_list = wait_and_send_ids_conf->mutable_id_list();
    FOR_RANGE(int32_t, i, 0, Global<JobName2JobId>::Get()->size()) { id_list->Add(); }
    HashSet<int64_t> unique_check;
    for (const auto& pair : *Global<JobName2JobId>::Get()) {
      int64_t job_id = pair.second;
      CHECK_OR_RETURN(unique_check.insert(job_id).second);
      const auto& cs_idx = Global<CriticalSectionDesc>::Get()->CriticalSectionIds4JobId(job_id);
      *id_list->Mutable(job_id)->mutable_value() = {cs_idx.begin(), cs_idx.end()};
    }
    JUST(job_builder.AddOp(parallel_conf, wait_and_send_ids_op_conf));
  }
  const int64_t num_critial_sections = Global<CriticalSectionDesc>::Get()->CriticalSectionNum();
  std::vector<std::map<int64_t, std::string>> cb_sink_tick_op_names;
  identity_tick_op_names->resize(num_critial_sections);
  cb_sink_tick_op_names.resize(num_critial_sections);
  const int64_t num_machines = Global<ResourceDesc, ForSession>::Get()->TotalMachineNum();
  const Range machine_id_range(0, num_machines);
  JUST(machine_id_range.ForEachSubRange(1, [&](const Range& sub_range) -> Maybe<void> {
    const auto& in_lbn = wait_and_send_ids_op_conf.name() + "/out";
    lock_back_edges->push_back(*JUST(MakeMainJobComponent(
        in_lbn, sub_range, &job_builder, identity_tick_op_names, &cb_sink_tick_op_names)));
    return Maybe<void>::Ok();
  }));
  OperatorConf callback_notify_esac_op_conf;
  {
    callback_notify_esac_op_conf.set_name(std::string("System-Main-Esac_") + NewUniqueId());
    auto* callback_notify_esac_conf = callback_notify_esac_op_conf.mutable_esac_conf();
    JUST(MakeCallbackNotifierSinkTick(
        machine_id_range, cb_sink_tick_op_names, &job_builder,
        [&](const std::string& lbn) { callback_notify_esac_conf->add_in(lbn); }));
    callback_notify_esac_conf->set_out("out");
    callback_notify_esac_conf->set_data_type(DataType::kInt32);
    JUST(job_builder.AddOp(parallel_conf, callback_notify_esac_op_conf));
  }
  OperatorConf callback_notify_op_conf;
  {
    callback_notify_op_conf.set_name(std::string("System-Main-CallbackNotify_") + NewUniqueId());
    auto* callback_notify_conf = callback_notify_op_conf.mutable_callback_notify_conf();
    callback_notify_conf->set_in(callback_notify_esac_op_conf.name() + "/out");
    auto* buffer_names = callback_notify_conf->mutable_callback_buffer_name();
    FOR_RANGE(int64_t, i, 0, Global<JobName2JobId>::Get()->size()) { buffer_names->Add(); }
    for (const auto& pair : *Global<JobName2JobId>::Get()) {
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
    const auto& op_name = kernel_conf.op_attribute().op_conf().name();
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
  CHECK(Global<JobName2JobId>::Get()->emplace(job_name, job_id).second);
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
      const std::string& op_name = kernel_conf.op_attribute().op_conf().name();
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
  auto* critical_section_desc = Global<CriticalSectionDesc>::Get();
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

void MakePullJob(const std::string& job_name, const std::string& op_name,
                 const ParallelBlobConf& parallel_blob_conf, Job* job) {
  auto* flag_name2flag_value = job->mutable_job_conf()->mutable_flag_name2flag_value();
  (*flag_name2flag_value)["__is_user_function__"].set_at_bool(false);
  auto* op_name2job_name =
      Global<InterUserJobInfo>::Get()->mutable_output_or_var_op_name2pull_job_name();
  CHECK(op_name2job_name->find(op_name) == op_name2job_name->end());
  (*op_name2job_name)[op_name] = job_name;
  DataType data_type;
  JobBuilder job_builder(job);
  OperatorConf input_op_conf;
  {
    input_op_conf.set_name(op_name);
    auto* input_conf = input_op_conf.mutable_input_conf();
    input_conf->set_out("out");
    auto* blob_conf = input_conf->mutable_blob_conf();
    InterfaceOpUtil::InitBlobConf(blob_conf, parallel_blob_conf);
    data_type = blob_conf->data_type();
    job_builder.AddOps(parallel_blob_conf.parallel_conf(), {input_op_conf});
  }
  OperatorConf foreign_output_op_conf;
  {
    foreign_output_op_conf.set_name(std::string("System-Pull-ForeignOutput_") + NewUniqueId());
    auto* foreign_output_conf = foreign_output_op_conf.mutable_foreign_output_conf();
    foreign_output_conf->set_in(input_op_conf.name() + "/out");
    foreign_output_conf->set_ofblob_buffer_name(GetForeignOutputBufferName(job_name));
    ParallelConf parallel_conf;
    parallel_conf.set_device_tag("cpu");
    parallel_conf.add_device_name("0:0");
    job_builder.AddOps(parallel_conf, {foreign_output_op_conf});
  }
  auto* job_conf = job->mutable_job_conf();
  job_conf->set_job_name(job_name);
  job_conf->mutable_predict_conf();
  job_conf->set_total_batch_num(1);
  job_conf->set_default_data_type(data_type);
}

void MakePushJob(const std::string& job_name, const std::string& op_name,
                 const ParallelBlobConf& parallel_blob_conf, Job* job) {
  auto* flag_name2flag_value = job->mutable_job_conf()->mutable_flag_name2flag_value();
  (*flag_name2flag_value)["__is_user_function__"].set_at_bool(false);
  auto* op_name2job_name =
      Global<InterUserJobInfo>::Get()->mutable_input_or_var_op_name2push_job_name();
  CHECK(op_name2job_name->find(op_name) == op_name2job_name->end());
  (*op_name2job_name)[op_name] = job_name;
  DataType data_type;
  JobBuilder job_builder(job);
  OperatorConf foreign_input_op_conf;
  {
    foreign_input_op_conf.set_name(std::string("System-Push-ForeignInput_") + NewUniqueId());
    auto* foreign_input_conf = foreign_input_op_conf.mutable_foreign_input_conf();
    foreign_input_conf->set_out("out");
    foreign_input_conf->set_ofblob_buffer_name(GetForeignInputBufferName(job_name));
    auto* blob_conf = foreign_input_conf->mutable_blob_conf();
    InterfaceOpUtil::InitBlobConf(blob_conf, parallel_blob_conf);
    data_type = blob_conf->data_type();
    ParallelConf parallel_conf;
    parallel_conf.set_device_tag("cpu");
    parallel_conf.add_device_name("0:0");
    job_builder.AddOps(parallel_conf, {foreign_input_op_conf});
  }
  OperatorConf output_op_conf;
  {
    output_op_conf.set_name(op_name);
    auto* output_conf = output_op_conf.mutable_output_conf();
    output_conf->set_in(foreign_input_op_conf.name() + "/out");
    output_conf->set_out("out");
    InterfaceOpUtil::InitBlobConf(output_conf->mutable_blob_conf(), parallel_blob_conf);
    job_builder.AddOps(parallel_blob_conf.parallel_conf(), {output_op_conf});
  }
  auto* job_conf = job->mutable_job_conf();
  job_conf->set_job_name(job_name);
  job_conf->mutable_predict_conf();
  job_conf->set_total_batch_num(1);
  job_conf->set_default_data_type(data_type);
}

REGISTER_FUNCTION_CONFIG_DEF().Bool("__is_user_function__", true, "is user defined function");

Maybe<void> CompileAndMergePlanOnMaster(const PbRpf<Job>& conf_jobs, Plan* plan) {
  std::vector<std::shared_ptr<Job>> jobs(conf_jobs.size());
  FOR_RANGE(int, i, 0, jobs.size()) { jobs.at(i).reset(new Job(conf_jobs.Get(i))); }
  if (jobs.size() > 1) { CheckNonDistributeOptimizerAvailable(jobs); }
  if (GlobalProcessCtx::IsThisProcessMaster()) {
    HashMap<std::string, ParallelBlobConf> var_op_name2parallel_blob_conf;
    FilterOpName2ParallelBlobConf({OperatorConf::kVariableConf}, jobs,
                                  &var_op_name2parallel_blob_conf);
    auto AppendJob = [&](Job* job) {
      JobDesc job_desc(job->job_conf(), jobs.size());
      CHECK(!job_desc.Bool("__is_user_function__"));
      jobs.emplace_back(new Job(*job));
    };
    if (Global<const IOConf>::Get()->enable_legacy_model_io()) {
      if (Global<const IOConf>::Get()->enable_model_io_v2()) {
        MakeModelIoV2Jobs(jobs, var_op_name2parallel_blob_conf, AppendJob);
      } else {
        MakeModelIoJobs(jobs, var_op_name2parallel_blob_conf, AppendJob);
      }
    }
  }
  std::vector<std::shared_ptr<Job>> function_jobs;
  function_jobs.reserve(jobs.size());
  FOR_RANGE(int, i, 0, jobs.size()) {
    JobDesc job_desc(jobs.at(i)->job_conf(), i);
    if (job_desc.Bool("__is_user_function__")) { function_jobs.push_back(jobs.at(i)); }
  }
  if (GlobalProcessCtx::IsThisProcessMaster()) {
    HashMap<std::string, ParallelBlobConf> push_op_name2parallel_blob_conf;
    FilterOpName2ParallelBlobConf({OperatorConf::kInputConf}, function_jobs,
                                  &push_op_name2parallel_blob_conf);
    HashMap<std::string, ParallelBlobConf> pull_op_name2parallel_blob_conf;
    FilterOpName2ParallelBlobConf({OperatorConf::kReturnConf}, function_jobs,
                                  &pull_op_name2parallel_blob_conf);
    for (const auto& pair : push_op_name2parallel_blob_conf) {
      auto push_job = std::make_shared<Job>();
      MakePushJob(std::string("System-Push-") + pair.first, pair.first, pair.second,
                  push_job.get());
      jobs.emplace_back(push_job);
    }
    for (const auto& pair : pull_op_name2parallel_blob_conf) {
      auto pull_job = std::make_shared<Job>();
      MakePullJob(std::string("System-Pull-") + pair.first, pair.first, pair.second,
                  pull_job.get());
      jobs.emplace_back(pull_job);
    }
  }
  std::vector<Plan> sub_plans(jobs.size());
  FOR_RANGE(int64_t, i, 0, jobs.size()) {
    AddJobName2JobId(jobs.at(i)->job_conf().job_name(), i);
    auto scope = std::make_unique<GlobalJobDescScope>(jobs.at(i)->job_conf(), i);
    JUST(CompileCurJobOnMaster(jobs.at(i).get(), &sub_plans.at(i), true));
  }
  if (GlobalProcessCtx::IsThisProcessMaster()) {
    MergeSubPlanWithoutGenNetTopo(plan, sub_plans);
    InterJobMemSharingUtil::MergeMemReusedChunkBetweenUserJobs(function_jobs, plan);
    InterJobMemSharingUtil::MergeMemSharedInterfaceMemBlockBetweenJobs(jobs, plan);
    PlanUtil::SetForceInplaceMemBlock(plan);
    FinishGlobalCriticalSectionDesc(*plan, jobs.size());
    Plan main_plan;
    std::vector<std::map<int64_t, std::string>> identity_tick_op_names;
    {
      Job main_job;
      std::vector<ReentrantLockBackEdge> lock_back_edges;
      JUST(MakeMainJob(&main_job, &identity_tick_op_names, &lock_back_edges));
      AddJobName2JobId(main_job.job_conf().job_name(), jobs.size());
      JUST(CompileMainJob(&main_job, lock_back_edges, sub_plans.size(), &main_plan));
    }
    LinkMainPlan(plan, main_plan, identity_tick_op_names);
    PlanUtil::CleanUselessMemBlockAndCheckValid(plan);
    if (Global<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
      TeePersistentLogStream::Create("merged_plan")->Write(*plan);
      PlanUtil::ToDotFile(*plan, "/dot/merged_plan.dot");
    }
    PushPlan("merged_plan", *plan);
  } else {
    PullPlan("merged_plan", plan);
    if (Global<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
      TeePersistentLogStream::Create("merged_plan")->Write(*plan);
    }
  }
  OF_SESSION_BARRIER();
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> Oneflow::Init(const oneflow::JobSet& job_set) {
  OF_PROFILER_RANGE_GUARD("Oneflow::Init");
  // Runtime
  OF_PROFILER_RANGE_PUSH("CompileAndMergePlanOnMaster");
  JUST(CompileAndMergePlanOnMaster(job_set.job(), &plan_));
  OF_PROFILER_RANGE_POP();  // CompileAndMergePlanOnMaster
  if (GlobalProcessCtx::IsThisProcessMaster()) {
    runtime_buffers_scope_.reset(new RuntimeBuffersScope(plan_));
  }
  OF_PROFILER_RANGE_PUSH("new Runtime");
  runtime_.reset(new Runtime(plan_, GetMaxVal<size_t>(), false));
  OF_PROFILER_RANGE_POP();  // new Runtime
  return Maybe<void>::Ok();
}

Oneflow::~Oneflow() {
  if (GlobalProcessCtx::IsThisProcessMaster()) { runtime_buffers_scope_.reset(); }
  runtime_.reset();
  if (Global<Profiler>::Get() != nullptr) {
    Global<Profiler>::Get()->Profile(
        plan_, JoinPath(FLAGS_log_dir, ActEventLogger::act_event_bin_filename()));
  }
}

}  // namespace oneflow
