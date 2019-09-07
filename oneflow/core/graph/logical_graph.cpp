#include "oneflow/core/graph/logical_graph.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

LogicalGraph::LogicalGraph(const Job& job) : job_(job) {
  BuildFwStruct();
  MergeEdge();
  SetNodeDataLbi();
  ToDotWithAutoFilePath();
}

template<typename LogicalNodeType>
void LogicalGraph::ForEachLogicalNode(std::function<void(LogicalNodeType*)> func) {
  std::vector<LogicalNodeType*> valid_nodes;
  ForEachNode([&](LogicalNode* logical_node) {
    auto valid_node = dynamic_cast<LogicalNodeType*>(logical_node);
    if (valid_node != nullptr) { valid_nodes.push_back(valid_node); }
  });
  for (LogicalNodeType* valid_node : valid_nodes) { func(valid_node); }
}

void LogicalGraph::BuildFwStruct() {
  HashMap<std::string, std::vector<LogicalNode*>> op_name2nodes;
  NaiveBuildFwStruct(&op_name2nodes);
  ReplaceAllReduceFacades();
  LinkUnpackFw2PackFw(op_name2nodes);
}

void LogicalGraph::NaiveBuildFwStruct(
    HashMap<std::string, std::vector<LogicalNode*>>* op_name2nodes) {
  const DLNetConf& dlnet_conf = job_.net();
  const Placement& placement = job_.placement();
  HashMap<std::string, std::shared_ptr<ParallelDesc>> name2parallel_desc;
  for (const PlacementGroup& p_group : placement.placement_group()) {
    for (const std::string& op_name : p_group.op_set().op_name()) {
      CHECK(name2parallel_desc
                .emplace(op_name, std::make_shared<ParallelDesc>(p_group.parallel_conf()))
                .second);
    }
  }

  HashMap<LogicalBlobId, std::string> lbi2obn;
  HashMap<LogicalBlobId, LogicalNode*> lbi2producer;
  for (OperatorConf cur_op_conf : dlnet_conf.op()) {
    auto parallel_desc_ptr_it = name2parallel_desc.find(cur_op_conf.name());
    CHECK(parallel_desc_ptr_it != name2parallel_desc.end());
    const std::shared_ptr<ParallelDesc>& parallel_desc_ptr = parallel_desc_ptr_it->second;
    cur_op_conf.set_device_type(parallel_desc_ptr->device_type());
    std::shared_ptr<Operator> cur_op = ConstructOp(cur_op_conf);
    LogicalNode* cur_node = cur_op->NewProperLogicalNode();
    if (cur_node->TypeName() == "PackForward" || cur_node->TypeName() == "UnpackForward") {
      CHECK_EQ(0, GlobalJobDesc().job_conf().piece_size() % parallel_desc_ptr->parallel_num());
    }
    AddAllocatedNode(cur_node);
    cur_node->mut_op_vec() = {cur_op};
    cur_node->SoleOp()->FixParallelDesc(parallel_desc_ptr.get());
    cur_node->mut_parallel_desc() = parallel_desc_ptr;
    {
      const auto& name2shape = job_.helper().op_name2op_time_shape();
      const auto& op_time_shape_it = name2shape.find(cur_op->op_name());
      if (op_time_shape_it != name2shape.end()) {
        const auto& op_time_shape = op_time_shape_it->second;
        if (op_time_shape.has_out_blob_time_shape()) {
          cur_node->reset_out_blob_time_shape(new Shape(op_time_shape.out_blob_time_shape()));
        }
        if (op_time_shape.has_in_blob_fastest_time_shape()) {
          cur_node->reset_in_blob_fastest_time_shape(
              new Shape(op_time_shape.in_blob_fastest_time_shape()));
        }
      }
    }
    for (const std::string& obn : cur_node->SoleOp()->output_bns()) {
      const LogicalBlobId& lbi = cur_node->SoleOp()->BnInOp2Lbi(obn);
      CHECK(lbi2producer.emplace(lbi, cur_node).second);
      CHECK(lbi2obn.emplace(lbi, obn).second);
    }
    (*op_name2nodes)[cur_op->op_name()].push_back(cur_node);
  }
  ForEachNode([&](LogicalNode* cur_node) {
    for (const std::string& ibn : cur_node->SoleOp()->input_bns()) {
      const LogicalBlobId& lbi = cur_node->SoleOp()->BnInOp2Lbi(ibn);
      LogicalNode* pred_node = lbi2producer.at(lbi);
      if (pred_node == cur_node) { continue; }
      LogicalEdge* edge = NewEdge();
      edge->mut_lbis() = {lbi};
      UpdateEdge2Ibn(edge, ibn);
      UpdateEdge2Obn(edge, lbi2obn.at(lbi));
      Connect(pred_node, edge, cur_node);
    }
  });
  // set batch_dim_lbis_cnt
  HashSet<LogicalBlobId> batch_dim_lbis;
  for (const auto& pair : job_.helper().lbn2batch_axis()) {
    if (pair.second.has_value() == false) { continue; }
    CHECK(batch_dim_lbis.emplace(GenLogicalBlobId(pair.first)).second);
  }
  ForEachNode([&](LogicalNode* cur_node) {
    size_t consumed_batch_dim_lbis_cnt = 0;
    size_t produced_batch_dim_lbis_cnt = 0;
    for (const auto& op : cur_node->op_vec()) {
      for (const std::string& ibn : op->input_bns()) {
        consumed_batch_dim_lbis_cnt +=
            batch_dim_lbis.find(op->BnInOp2Lbi(ibn)) != batch_dim_lbis.end();
      }
      for (const std::string& obn : op->output_bns()) {
        produced_batch_dim_lbis_cnt +=
            batch_dim_lbis.find(op->BnInOp2Lbi(obn)) != batch_dim_lbis.end();
      }
    }
    cur_node->set_consumed_batch_dim_lbis_cnt(consumed_batch_dim_lbis_cnt);
    cur_node->set_produced_batch_dim_lbis_cnt(produced_batch_dim_lbis_cnt);
  });
}

void LogicalGraph::LinkUnpackFw2PackFw(
    const HashMap<std::string, std::vector<LogicalNode*>>& op_name2nodes) {
  ForEachLogicalNode<PackForwardLogicalNode>([&](PackForwardLogicalNode* pack_fw) {
    const std::string& unpack_name = pack_fw->SoleOp()->op_conf().pack_conf().related_unpack();
    auto it = op_name2nodes.find(unpack_name);
    CHECK(it != op_name2nodes.end());
    CHECK_EQ(1, it->second.size());
    UnpackForwardLogicalNode* unpack_fw =
        dynamic_cast<UnpackForwardLogicalNode*>(it->second.front());
    CHECK(unpack_fw);
    pack_fw->set_related_unpack(unpack_fw);
  });
}

void LogicalGraph::MergeEdge() {
  ForEachNode([](LogicalNode* node) {
    HashMap<LogicalNode*, std::vector<LogicalEdge*>> dst2edges;
    for (LogicalEdge* out_edge : node->out_edges()) {
      dst2edges[out_edge->dst_node()].push_back(out_edge);
    }
    for (const auto& pair : dst2edges) {
      std::vector<LogicalBlobId>& lbi_all = pair.second.at(0)->mut_lbis();
      FOR_RANGE(size_t, i, 1, pair.second.size()) {
        std::vector<LogicalBlobId>& lbi_i = pair.second.at(i)->mut_lbis();
        lbi_all.insert(lbi_all.end(), lbi_i.begin(), lbi_i.end());
        lbi_i.clear();
        DisConnect(pair.second.at(i));  // TODO: delete its memory ?
      }
    }
  });
}

void LogicalGraph::SetNodeDataLbi() {
  ForEachNode([](LogicalNode* node) {
    for (LogicalEdge* out_edge : node->out_edges()) {
      node->SetDataLbisTo(out_edge->dst_node(), out_edge->lbis());
    }
  });
}

bool LogicalGraph::MustHaveModelDiffAcc() {
  bool must_have_model_diff_acc = false;
  ForEachLogicalNode<ForwardLogicalNode>(
      [&must_have_model_diff_acc](ForwardLogicalNode* fw_logical) {
        if (must_have_model_diff_acc) { return; }
        if (fw_logical->TypeName() == "PackForward" || fw_logical->TypeName() == "UnpackForward"
            || fw_logical->TypeName() == "RepeatForward") {
          must_have_model_diff_acc = true;
          return;
        }
      });
  return must_have_model_diff_acc;
}

void LogicalGraph::AddAllReduce(LogicalNode* src, LogicalNode* dst) {
  std::shared_ptr<const ParallelDesc> src_pd = src->parallel_desc();
  std::shared_ptr<const ParallelDesc> dst_pd = dst->parallel_desc();
  CHECK_EQ(src_pd->parallel_num(), dst_pd->parallel_num());
  CHECK_EQ(src_pd->device_type(), dst_pd->device_type());
  if (GlobalJobDesc().enable_nccl() && src_pd->device_type() == DeviceType::kGPU) {
    if (src_pd->sorted_machine_ids().size() == 1
        || GlobalJobDesc().use_nccl_inter_node_communication()) {
      AddNcclAllReduce(src, dst);
    } else if (src_pd->device_num_of_each_machine() == 1) {
      AddReduceScatterAddGatherNodes(src, dst, ReduceRankCtx());
    } else {
      AddNcclReduceScatterAndAllGather(src, dst);
    }
  } else {
    AddReduceScatterAddGatherNodes(src, dst, ReduceRankCtx());
  }
}

void LogicalGraph::AddNcclReduceScatterAndAllGather(LogicalNode* src, LogicalNode* dst) {
  std::shared_ptr<const ParallelDesc> src_pd = src->parallel_desc();

  ReduceRankCtx rank_ctx = ReduceRankCtx().CtxWithScatter(src_pd->device_num_of_each_machine());

  OperatorConf nccl_reduce_scatter_op_conf{};
  nccl_reduce_scatter_op_conf.set_name("nccl_reduce_scatter_" + NewUniqueId());
  nccl_reduce_scatter_op_conf.set_device_type(src_pd->device_type());
  nccl_reduce_scatter_op_conf.mutable_nccl_reduce_scatter_conf();
  NcclReduceScatterLogicalNode* nccl_reduce_scatter_node = NewNode<NcclReduceScatterLogicalNode>();
  nccl_reduce_scatter_node->mut_op_vec() = {ConstructOp(nccl_reduce_scatter_op_conf)};
  nccl_reduce_scatter_node->mut_parallel_desc() = src_pd;
  nccl_reduce_scatter_node->mut_rank_ctx() = rank_ctx;
  Connect<LogicalNode>(src, NewEdge(), nccl_reduce_scatter_node);

  OperatorConf nccl_all_gather_op_conf{};
  nccl_all_gather_op_conf.set_name("nccl_all_gather_" + NewUniqueId());
  nccl_all_gather_op_conf.set_device_type(src_pd->device_type());
  nccl_all_gather_op_conf.mutable_nccl_all_gather_conf();
  NcclAllGatherLogicalNode* nccl_all_gather_node = NewNode<NcclAllGatherLogicalNode>();
  nccl_all_gather_node->mut_op_vec() = {ConstructOp(nccl_all_gather_op_conf)};
  nccl_all_gather_node->mut_parallel_desc() = src_pd;
  nccl_all_gather_node->mut_rank_ctx() = rank_ctx;
  Connect<LogicalNode>(nccl_all_gather_node, NewEdge(), dst);

  AddReduceScatterAddGatherNodes(nccl_reduce_scatter_node, nccl_all_gather_node, rank_ctx);
}

void LogicalGraph::AddNcclAllReduce(LogicalNode* src, LogicalNode* dst) {
  std::shared_ptr<const ParallelDesc> src_pd = src->parallel_desc();
  OperatorConf nccl_all_reduce_op_conf{};
  nccl_all_reduce_op_conf.set_name("nccl_all_reduce_" + NewUniqueId());
  nccl_all_reduce_op_conf.set_device_type(src_pd->device_type());
  nccl_all_reduce_op_conf.mutable_nccl_all_reduce_conf();
  NcclAllReduceLogicalNode* nccl_all_reduce_node = NewNode<NcclAllReduceLogicalNode>();
  nccl_all_reduce_node->mut_op_vec() = {ConstructOp(nccl_all_reduce_op_conf)};
  nccl_all_reduce_node->mut_parallel_desc() = src_pd;
  nccl_all_reduce_node->mut_rank_ctx() = ReduceRankCtx().CtxWithScatter(src_pd->parallel_num());
  Connect<LogicalNode>(src, NewEdge(), nccl_all_reduce_node);
  Connect<LogicalNode>(nccl_all_reduce_node, NewEdge(), dst);
}

void LogicalGraph::AddReduceScatterAddGatherNodes(LogicalNode* src, LogicalNode* dst,
                                                  const ReduceRankCtx& prev_rank_ctx) {
  std::shared_ptr<const ParallelDesc> src_pd = src->parallel_desc();

  int64_t segment_count =
      prev_rank_ctx.TotalSegmentCount() == 1
          ? (src_pd->device_num_of_each_machine() == 1 ? src_pd->sorted_machine_ids().size()
                                                       : src_pd->device_num_of_each_machine())
          : src_pd->sorted_machine_ids().size();

  ReduceRankCtx current_rank_ctx = prev_rank_ctx.CtxWithScatter(segment_count);
  ReduceScatterLogicalNode* reduce_scatter_node = NewNode<ReduceScatterLogicalNode>();
  OperatorConf reduce_scatter_op_conf{};
  reduce_scatter_op_conf.set_name("reduce_scatter_" + NewUniqueId());
  reduce_scatter_op_conf.set_device_type(src_pd->device_type());
  reduce_scatter_op_conf.mutable_reduce_scatter_conf();
  reduce_scatter_node->mut_op_vec() = {ConstructOp(reduce_scatter_op_conf)};
  reduce_scatter_node->mut_parallel_desc() = src_pd;
  reduce_scatter_node->mut_rank_ctx() = current_rank_ctx;
  Connect<LogicalNode>(src, NewEdge(), reduce_scatter_node);

  ReduceAddLogicalNode* reduce_add_node = NewNode<ReduceAddLogicalNode>();
  OperatorConf reduce_add_op_conf{};
  reduce_add_op_conf.set_name("reduce_add_" + NewUniqueId());
  reduce_add_op_conf.set_device_type(src_pd->device_type());
  reduce_add_op_conf.mutable_reduce_add_conf();
  reduce_add_node->mut_op_vec() = {ConstructOp(reduce_add_op_conf)};
  reduce_add_node->mut_parallel_desc() = src_pd;
  reduce_add_node->mut_rank_ctx() = current_rank_ctx;
  Connect<LogicalNode>(reduce_scatter_node, NewEdge(), reduce_add_node);

  ReduceGatherLogicalNode* reduce_gather_node = NewNode<ReduceGatherLogicalNode>();
  OperatorConf reduce_gather_op_conf{};
  reduce_gather_op_conf.set_name("reduce_gather_" + NewUniqueId());
  reduce_gather_op_conf.set_device_type(src_pd->device_type());
  reduce_gather_op_conf.mutable_reduce_gather_conf();
  reduce_gather_node->mut_op_vec() = {ConstructOp(reduce_gather_op_conf)};
  reduce_gather_node->mut_parallel_desc() = src_pd;
  reduce_gather_node->mut_rank_ctx() = current_rank_ctx;

  if (current_rank_ctx.TotalSegmentCount() == src_pd->parallel_num()) {
    Connect<LogicalNode>(reduce_add_node, NewEdge(), reduce_gather_node);
  } else {
    AddReduceScatterAddGatherNodes(reduce_add_node, reduce_gather_node, current_rank_ctx);
  }

  Connect<LogicalNode>(reduce_gather_node, NewEdge(), dst);
}

void LogicalGraph::ForEachNecessaryCtrlEdge(
    const std::function<void(const LogicalNode*, const LogicalNode*, int64_t)>& Handler) const {
  HashMap<std::string, const LogicalNode*> op_name2node;
  ForEachNode([&](LogicalNode* node) {
    for (const auto& op : node->op_vec()) {
      CHECK(op_name2node.emplace(op->op_name(), node).second);
    }
  });
  auto IsReachable = MakePredicatorIsReachable();
  ForEachNode([&](LogicalNode* dst) {
    for (const auto& op : dst->op_vec()) {
      for (const auto& ctrl_in_op_name : op->op_conf().ctrl_in_op_name()) {
        const LogicalNode* src = op_name2node.at(ctrl_in_op_name);
        CHECK(!IsReachable(dst, src));
        if (!IsReachable(src, dst)) {
          CHECK(src->parallel_desc()->EqualsIgnoringPolicyAndDeviceType(*dst->parallel_desc()));
          const Shape* src_time_shape = src->out_blob_time_shape();
          if (src_time_shape == nullptr) { src_time_shape = src->in_blob_fastest_time_shape(); }
          CHECK_NOTNULL(src_time_shape);
          const Shape* dst_time_shape = dst->in_blob_fastest_time_shape();
          if (dst_time_shape == nullptr) { dst_time_shape = dst->out_blob_time_shape(); }
          CHECK_NOTNULL(dst_time_shape);
          CHECK(src_time_shape->elem_cnt() == dst_time_shape->elem_cnt()
                || src_time_shape->Containing(*dst_time_shape));
          CHECK_EQ(src_time_shape->elem_cnt() % dst_time_shape->elem_cnt(), 0);
          int64_t regst_desc_num = src_time_shape->elem_cnt() / dst_time_shape->elem_cnt();
          Handler(src, dst, regst_desc_num);
        }
      }
    }
  });
}

void LogicalGraph::ReplaceAllReduceFacades() {
  ForEachLogicalNode<AllReduceFacadeLogicalNode>([this](AllReduceFacadeLogicalNode* facade_node) {
    CHECK_EQ(facade_node->in_edges().size(), 1);
    CHECK_EQ(facade_node->out_edges().size(), 1);
    LogicalNode* src = facade_node->SoleInEdge()->src_node();
    LogicalNode* dst = facade_node->SoleOutEdge()->dst_node();
    DisConnect(facade_node->SoleInEdge());
    DisConnect(facade_node->SoleOutEdge());
    DeleteNode(facade_node);
    AddAllReduce(src, dst);
    Operator* all_reduce_ending_op = dst->SoleInEdge()->src_node()->SoleOp().get();
    const LogicalBlobId& ending_lbi =
        all_reduce_ending_op->BnInOp2Lbi(all_reduce_ending_op->SoleObn());
    *dst->SoleOp()->MutBnInOp2Lbi(dst->SoleOp()->SoleIbn()) = ending_lbi;
  });
}

void LogicalGraph::UpdateEdge2Ibn(const LogicalEdge* edge, const std::string& ibn) {
  if (!ibn.empty()) { edge2ibn_[edge] = ibn; }
}

void LogicalGraph::UpdateEdge2Obn(const LogicalEdge* edge, const std::string& obn) {
  if (!obn.empty()) { edge2obn_[edge] = obn; }
}

}  // namespace oneflow
