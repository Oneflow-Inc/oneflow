#include "oneflow/core/job/compiler.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/job/cudnn_conv_ctx_cache_scope.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job_completer/job_completer.h"

namespace oneflow {

void Compiler::GenNetTopo(Plan* plan) const {
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

void Compiler::Compile(Job* job, Plan* plan, bool need_job_complete) const {
  auto cudnn_conv_ctx_cache_scope = std::make_unique<CudnnConvCtxCacheScope>();
  const JobDesc& job_desc = GlobalJobDesc();
  if (need_job_complete) { JobCompleter().Complete(job); }
  TeePersistentLogStream::Create(StrCat("optimized_job", job_desc.job_id()))->Write(*job);
  Global<OpGraph>::New(*job);
  Global<OpGraph>::Get()->ToDotWithFilePath("optimized_dlnet_" + std::to_string(job_desc.job_id())
                                            + "_op_graph.dot");
  auto logical_gph = std::make_unique<LogicalGraph>(*job);
  auto task_gph = std::make_unique<TaskGraph>(std::move(logical_gph));
  using std::placeholders::_1;
  task_gph->ForEachNode(std::bind(&TaskNode::ProduceAllRegstsAndBindEdges, _1));
  task_gph->ForEachNode(std::bind(&TaskNode::ConsumeAllRegsts, _1));
  task_gph->ForEachNode(std::bind(&TaskNode::PinConsumedRegst, _1));
  task_gph->MdUpdtDelayedTopoForEachNode(&TaskNode::Build);
  task_gph->RemoveEmptyRegsts();
  task_gph->AddOrderingCtrlEdgeInSameChain();
  // TODO: update method for fw bw split
  // if (job_desc.IsTrain() && job_desc.enable_mem_sharing()) {
  //   task_gph->EnableMemSharingAfterAllManualSetForMdUpdt();  // must last mem shared manual set
  // }
  if (job_desc.enable_inplace_in_reduce_struct()) {
    task_gph->EnableInplaceMemSharingInReduceStruct();
  }

  if (job_desc.enable_inplace()) {
    auto IsReachable = Global<OpGraph>::Get()->MakePredicatorIsOpNameDataOrCtrlReachable();
    task_gph->EnableInplaceMemSharing(IsReachable);
  }
  // TODO: update method for fw bw split
  // if (job_desc.IsTrain()) { task_gph->AddOrderCtrlEdgeBetweenCopyAndMdUpdt(); }
  task_gph->MdUpdtDelayedTopoForEachNode(&TaskNode::InferTimeShapeIfMeaningful);
  // TODO: update method for fw bw split
  // if (job_desc.IsTrain()) { task_gph->AddReduceNoBwForwardNodeOverlapingCtrlEdges(); }

  task_gph->ForEachNode([&](TaskNode* task_node) {
    if (task_node->IsMeaningLess()) { return; }
    task_node->ToProto(plan->mutable_task()->Add());
  });
  {
    auto* job_id2job_conf = plan->mutable_job_confs()->mutable_job_id2job_conf();
    (*job_id2job_conf)[GlobalJobDesc().job_id()] = GlobalJobDesc().job_conf();
  }
  // TODO: fix .dot generate
  // GenNetTopo(plan);
  Global<OpGraph>::Delete();
}

}  // namespace oneflow
