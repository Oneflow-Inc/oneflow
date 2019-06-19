#include "oneflow/core/job/compiler.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/device/cudnn_conv_ctx_cache.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job_completer/job_completer.h"

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

Job ConvertJobConf2Job(const JobConf& job_conf) {
  Job job;
  *job.mutable_net() = job_conf.net();
  *job.mutable_resource() = job_conf.resource();
  *job.mutable_placement() = job_conf.placement();
  *job.mutable_sbp_conf() = job_conf.sbp_conf();
  *job.mutable_other() = job_conf.other();
  return job;
}

}  // namespace

Plan Compiler::Compile() {
  Plan plan = DoCompile();
  return plan;
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
  Job job = ConvertJobConf2Job(job_desc->job_conf());
  JobCompleter().Complete(&job);
  TeePersistentLogStream::Create("optimized_job")->Write(job);
  Global<OpGraph>::New(job);
  Global<OpGraph>::Get()->ToDotWithFilePath("optimized_dlnet_op_graph.dot");
  auto logical_gph = std::make_unique<LogicalGraph>(job);
  int64_t total_mbn_num = logical_gph->total_mbn_num();
  auto task_gph = std::make_unique<TaskGraph>(std::move(logical_gph));
  using std::placeholders::_1;
  task_gph->ForEachNode(std::bind(&TaskNode::ProduceAllRegstsAndBindEdges, _1));
  task_gph->ForEachNode(std::bind(&TaskNode::ConsumeAllRegsts, _1));
  task_gph->ForEachNode(std::bind(&TaskNode::PinConsumedRegst, _1));
  task_gph->MdUpdtDelayedTopoForEachNode(&TaskNode::Build);
  if (job_desc->other_conf().predict_conf().has_tmp_split_fw_bw_train_conf()) {
    task_gph->AddReduceSequenceCtrlEdges();
    // TODO: update method for fw bw split
    // task_gph->AddMdUpdtCtrlEdgesWithinReduceSplitNode();
  }
  task_gph->RemoveEmptyRegsts();
  task_gph->AddOrderingCtrlEdgeInSameChain();
  task_gph->EnableMemSharingInReduceStruct();
  // TODO: update method for fw bw split
  // if (job_desc->IsTrain() && job_desc->enable_mem_sharing()) {
  //   task_gph->EnableMemSharingAfterAllManualSetForMdUpdt();  // must last mem shared manual set
  // }
  if (job_desc->enable_inplace()) {
    auto IsReachable = Global<OpGraph>::Get()->MakePredicatorIsLbiAllConsumersReachableToOpName();
    task_gph->EnableInplaceMemSharing(IsReachable);
  }
  // TODO: update method for fw bw split
  // if (job_desc->IsTrain()) { task_gph->AddOrderCtrlEdgeBetweenCopyAndMdUpdt(); }
  task_gph->MdUpdtDelayedTopoForEachNode(&TaskNode::InferTimeShapeIfMeaningful);
  // TODO: update method for fw bw split
  // if (job_desc->IsTrain() && job_desc->enable_mem_sharing()) {
  //   task_gph->EnableMemSharingInVariableOp();
  // }
  // TODO: update method for fw bw split
  // if (job_desc->IsTrain()) { task_gph->AddReduceNoBwForwardNodeOverlapingCtrlEdges(); }

  Plan plan;
  task_gph->ForEachNode([&](TaskNode* task_node) {
    if (task_node->IsMeaningLess()) { return; }
    task_node->ToProto(plan.mutable_task()->Add());
  });
  plan.set_total_mbn_num(total_mbn_num);
  GenNetTopo(&plan);
  ToDotFile(plan, "/dot/plan.dot");
  Global<OpGraph>::Delete();
#ifdef WITH_CUDA
  Global<CudnnConvCtxCache>::Delete();
#endif
  return plan;
}

}  // namespace oneflow
