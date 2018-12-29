#include "oneflow/core/job/compiler.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/device/cudnn_conv_ctx_cache.h"

namespace oneflow {

namespace {

void ToDotFile(const Plan& plan, const std::string& filepath) {
  auto log_stream = TeePersistentLogStream::Create(filepath);
  log_stream << "digraph {\n";
  HashSet<int64_t> regst_desc_ids;
  for (const TaskProto& task_proto : plan.task()) {
    log_stream << "task" << std::to_string(task_proto.task_id()) << "[label=\""
               << std::to_string(task_proto.task_id()) << "\\n"
               << std::to_string(task_proto.machine_id()) << ":"
               << std::to_string(task_proto.thrd_id()) << ":"
               << std::to_string(task_proto.parallel_ctx().parallel_id())
               << "\", shape=ellipse, style=\"rounded,filled\", "
                  "colorscheme=set312, color="
               << task_type2color.at(task_proto.task_type()) << "];\n";
    for (const auto& pair : task_proto.produced_regst_desc()) {
      regst_desc_ids.insert(pair.second.regst_desc_id());
    }
  }
  for (const int64_t regst_task_id : regst_desc_ids) {
    log_stream << "regst_desc" << std::to_string(regst_task_id) << "[label=\""
               << std::to_string(regst_task_id) << "\", shape=box];\n";
  }
  for (const TaskProto& task_proto : plan.task()) {
    for (const auto& pair : task_proto.produced_regst_desc()) {
      log_stream << "task" << std::to_string(task_proto.task_id()) << "->regst_desc"
                 << std::to_string(pair.second.regst_desc_id()) << "[label=\"" << pair.first
                 << "\"];\n";
    }
    for (const auto& pair : task_proto.consumed_regst_desc_id()) {
      for (int64_t regst_desc_id : pair.second.regst_desc_id()) {
        log_stream << "regst_desc" << std::to_string(regst_desc_id) << "->task"
                   << std::to_string(task_proto.task_id()) << "[label=\"" << pair.first << "\"];\n";
      }
    }
  }
  log_stream << "}\n";
}

bool IsNcclTask(const TaskProto& task) {
  switch (task.task_type()) {
    case TaskType::kNcclAllGather:
    case TaskType::kNcclAllReduce:
    case TaskType::kNcclReduceScatter: return true;
    default: return false;
  }
}

bool CanTaskReuseNcclComm(const TaskProto& task) {
  return task.task_type() == TaskType::kNcclAllReduce;
}

std::string GenNcclGroupKey(const std::vector<int64_t>& sorted_thrd_ids) {
  std::ostringstream oss;
  for (const int64_t thrd_id : sorted_thrd_ids) {
    oss << std::setfill('0') << std::setw(sizeof(thrd_id) * 2) << std::hex << thrd_id;
  }
  return oss.str();
}

bool IsTaskOnGpuDevice(const TaskProto& task) {
  return Global<IDMgr>::Get()->GetDeviceTypeFromThrdId(task.thrd_id()) == DeviceType::kGPU;
}

}  // namespace

Plan Compiler::Compile() {
  Plan plan = DoCompile();
  return plan;
}

void Compiler::GenNcclTopo(Plan* plan) {
  HashMap<int64_t, std::vector<const TaskProto*>> rank_set2nccl_tasks;
  HashMap<int64_t, bool> rank_set2reuse_nccl_comm;
  const bool enable_reuse_nccl_communicator =
      Global<JobDesc>::Get()->enable_reuse_nccl_communicator();
  for (const TaskProto& task : plan->task()) {
    if (!IsNcclTask(task)) { continue; }
    if (enable_reuse_nccl_communicator) { CHECK(task.task_type() == TaskType::kNcclAllReduce); }
    CHECK(IsTaskOnGpuDevice(task));
    CHECK(task.has_parallel_ctx());
    CHECK(task.parallel_ctx().has_rank_ctx());
    const int64_t rank_set_id = task.parallel_ctx().rank_ctx().rank_set_id();
    if (!rank_set2nccl_tasks[rank_set_id].empty()) {
      const TaskProto* first = rank_set2nccl_tasks[rank_set_id].front();
      CHECK_EQ(first->task_type(), task.task_type());
      CHECK_EQ(first->parallel_ctx().rank_ctx().rank_num(),
               task.parallel_ctx().rank_ctx().rank_num());
    }
    if (rank_set2reuse_nccl_comm.find(rank_set_id) == rank_set2reuse_nccl_comm.end()) {
      rank_set2reuse_nccl_comm[rank_set_id] = CanTaskReuseNcclComm(task);
    } else {
      CHECK_EQ(rank_set2reuse_nccl_comm[rank_set_id], CanTaskReuseNcclComm(task));
    }
    rank_set2nccl_tasks[rank_set_id].push_back(&task);
  }
  for (auto& pair : rank_set2nccl_tasks) {
    std::vector<const TaskProto*>& tasks = pair.second;
    auto CompareRankId = [](const TaskProto* lhs, const TaskProto* rhs) -> bool {
      return lhs->parallel_ctx().rank_ctx().rank_id() < lhs->parallel_ctx().rank_ctx().rank_id();
    };
    std::sort(tasks.begin(), tasks.end(), CompareRankId);
    CHECK_EQ(tasks.size(), tasks.front()->parallel_ctx().rank_ctx().rank_num());
    FOR_RANGE(size_t, i, 0, tasks.size()) {
      CHECK_EQ(tasks.at(i)->parallel_ctx().rank_ctx().rank_id(), i);
    }
  }
  HashMap<int64_t, int64_t> task_id2comm_desc_id;
  std::map<std::string, const NcclCommGroup*> nccl_group_key2nccl_group;
  NcclTopo* nccl_topo = plan->mutable_nccl_topo();
  int64_t next_nccl_comm_id = 0;
  auto GetOrCreateNcclGroup = [&](const std::vector<int64_t>& sorted_thrd_ids,
                                  const bool reuse) -> const NcclCommGroup* {
    const std::string key = GenNcclGroupKey(sorted_thrd_ids);
    if (reuse) {
      const auto it = nccl_group_key2nccl_group.find(key);
      if (it != nccl_group_key2nccl_group.end()) { return it->second; }
    }
    NcclCommGroup* group = nccl_topo->mutable_group()->Add();
    group->set_id(nccl_topo->group().size());
    FOR_RANGE(int32_t, i, 0, sorted_thrd_ids.size()) {
      NcclCommDesc* comm = group->mutable_comm_desc()->Add();
      comm->set_id(++next_nccl_comm_id);
      comm->set_global_thrd_id(sorted_thrd_ids[i]);
      comm->set_rank_id(i);
    }
    if (reuse) { nccl_group_key2nccl_group[key] = group; }
    return group;
  };
  for (const auto& pair : rank_set2nccl_tasks) {
    const std::vector<const TaskProto*>& tasks = pair.second;
    std::vector<int64_t> thrd_ids(tasks.size());
    std::transform(tasks.cbegin(), tasks.cend(), thrd_ids.begin(),
                   [](const TaskProto* task) -> int64_t {
                     return Global<IDMgr>::Get()->GlobalThrdId4TaskId(task->task_id());
                   });
    const NcclCommGroup* group = GetOrCreateNcclGroup(
        thrd_ids, enable_reuse_nccl_communicator && rank_set2reuse_nccl_comm[pair.first]);
    CHECK_EQ(group->comm_desc_size(), tasks.size());
    FOR_RANGE(int32_t, i, 0, tasks.size()) {
      task_id2comm_desc_id.emplace(tasks[i]->task_id(), group->comm_desc(i).id());
    }
  }
  *nccl_topo->mutable_task_id2comm_desc_id() = HashMap2PbMap(task_id2comm_desc_id);
}

void Compiler::GenNetTopo(Plan* plan) {
  HashMap<int64_t, int64_t> rid2mid;
  HashMap<int64_t, int64_t> tid2mid;
  std::map<int64_t, std::set<int64_t>> net_topo;

  for (const TaskProto& task_proto : plan->task()) {
    for (const auto& regst_desc_it : task_proto.produced_regst_desc()) {
      rid2mid.emplace(regst_desc_it.second.regst_desc_id(), task_proto.machine_id());
    }
    CHECK(tid2mid.emplace(task_proto.task_id(), task_proto.machine_id()).second);
  }

  for (const TaskProto& task_proto : plan->task()) {
    for (const auto& regst_desc_it : task_proto.produced_regst_desc()) {
      int64_t rid = regst_desc_it.second.regst_desc_id();
      auto rid2mid_it = rid2mid.find(rid);
      CHECK(rid2mid_it != rid2mid.end());
      int64_t producer_mid = rid2mid_it->second;
      for (int64_t consumer_task_id : regst_desc_it.second.consumer_task_id()) {
        auto tid2mid_it = tid2mid.find(consumer_task_id);
        CHECK(tid2mid_it != tid2mid.end());
        int64_t consumer_mid = tid2mid_it->second;
        net_topo[producer_mid].insert(consumer_mid);
        net_topo[consumer_mid].insert(producer_mid);
      }
    }
  }

  HashMap<int64_t, MachineIds> std_net_topo;
  NetTopo& pb_net_topo = *(plan->mutable_net_topo());
  for (auto& pair : net_topo) {
    int64_t src_mid = pair.first;
    if (pair.second.count(src_mid)) { pair.second.erase(src_mid); }
    std::vector<int64_t> peer_mids(pair.second.begin(), pair.second.end());
    MachineIds pb_mids;
    *(pb_mids.mutable_machine_id()) = StdVec2PbRf<int64_t>(peer_mids);
    CHECK(std_net_topo.emplace(src_mid, pb_mids).second);
  }
  *(pb_net_topo.mutable_peer_machine_ids()) = HashMap2PbMap(std_net_topo);
}

Plan Compiler::DoCompile() {
#ifdef WITH_CUDA
  Global<CudnnConvCtxCache>::New();
#endif
  const JobDesc* job_desc = Global<JobDesc>::Get();
  auto logical_gph = std::make_unique<LogicalGraph>(job_desc->IsTrain());
  int64_t total_mbn_num = logical_gph->total_mbn_num();
  auto task_gph = std::make_unique<TaskGraph>(std::move(logical_gph));
  using std::placeholders::_1;
  task_gph->ForEachNode(std::bind(&TaskNode::ProduceAllRegstsAndBindEdges, _1));
  task_gph->ForEachNode(std::bind(&TaskNode::ConsumeAllRegsts, _1));
  task_gph->ForEachNode(std::bind(&TaskNode::PinConsumedRegst, _1));
  task_gph->MdUpdtDelayedTopoForEachNode(&TaskNode::Build);
  if (job_desc->IsTrain()) {
    task_gph->AddReduceSequenceCtrlEdges();
    task_gph->AddMdUpdtCtrlEdgesWithinReduceSplitNode();
  }
  task_gph->RemoveEmptyRegsts();
  task_gph->AddOrderingCtrlEdgeInSameChain();
  if (job_desc->IsTrain() && job_desc->enable_mem_sharing()) {
    task_gph->EnableMemSharingInReduceStruct();
    task_gph->EnableMemSharingAfterAllManualSetForMdUpdt();  // must last mem shared manual set
  }
  if (job_desc->IsTrain()) { task_gph->AddOrderCtrlEdgeBetweenCopyAndMdUpdt(); }
  if (job_desc->IsTrain()) { task_gph->RmUselessConsumeRelationshipBetweenFwBw(); }
  task_gph->MdUpdtDelayedTopoForEachNode(&TaskNode::InferTimeShapeIfMeaningful);
  if (job_desc->IsTrain() && job_desc->enable_mem_sharing()) {
    task_gph->EnableMemSharingInVariableOp();
  }
  if (job_desc->IsTrain()) { task_gph->AddReduceNoBwForwardNodeOverlapingCtrlEdges(); }

  Plan plan;
  task_gph->ForEachNode([&](TaskNode* task_node) {
    if (task_node->IsMeaningLess()) { return; }
    task_node->ToProto(plan.mutable_task()->Add());
  });
  plan.set_total_mbn_num(total_mbn_num);
  GenNetTopo(&plan);
  GenNcclTopo(&plan);
  ToDotFile(plan, "/dot/plan.dot");
#ifdef WITH_CUDA
  Global<CudnnConvCtxCache>::Delete();
#endif
  return plan;
}

}  // namespace oneflow
