#include "oneflow/core/job_completer/non_distributed_optimizer_pass.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

namespace {

int64_t GetSoleOutBlobSize(const OpNode* node) {
  const BlobDesc& blob_desc =
      node->LogicalBlobDesc4Lbi(node->op().BnInOp2Lbi(node->op().SoleObn()));
  return blob_desc.shape().elem_cnt() * GetSizeOfDataType(blob_desc.data_type());
}

ParallelConf NonDistributedParallelConf4ParallelId(const ParallelDesc& pd,
                                                   const int64_t parallel_id) {
  std::string device_name;
  device_name += std::to_string(pd.MachineIdForParallelId(parallel_id));
  if (pd.device_type() == DeviceType::kGPU) {
    device_name += ":gpu:";
  } else if (pd.device_type() == DeviceType::kCPU) {
    device_name += ":cpu:";
  } else {
    UNIMPLEMENTED();
  }
  device_name += std::to_string(pd.DeviceIdForParallelId(parallel_id));
  ParallelConf parallel_conf;
  *parallel_conf.mutable_device_name()->Add() = device_name;
  return parallel_conf;
}

}  // namespace

void NonDistributedOptimizerPass::Apply(const OpGraph& op_graph, JobBuilder* builder) {
  HashMap<ParallelDesc, HashMap<const OpNode*, std::vector<const OpNode*>>> pd2last_node2node_seqs;
  HashMap<const OpNode*, OperatorConf> op_node2op_conf;
  HashMap<const OpNode*, int64_t> last_node2model_size;
  HashMap<const OpNode*, int64_t> op_node2topo_order;
  HashMap<ParallelDesc, std::vector<const OpNode*>> pd2last_nodes;
  HashMap<const OpNode*, int64_t> last_node2parallel_id;
  HashMap<const OpNode*, int64_t> last_node2order;
  int64_t node_cnt = 0;
  op_graph.TopoForEachNode([&](const OpNode* node) {
    op_node2topo_order[node] = node_cnt;
    node_cnt += 1;
  });
  op_graph.ForEachNode([&](const OpNode* node) {
    if (!node->op().op_conf().has_variable_conf()) { return; }
    std::vector<const OpNode*> op_seq_without_batch_dim;
    const OpNode* cur_node = node;
    while (cur_node != nullptr) {
      if (cur_node->parallel_desc().device_type() != DeviceType::kGPU) { break; }
      if (cur_node != node) {
        if (cur_node->in_edges().size() > 1) { break; }
        if (cur_node->op().input_bns().size() != 1) { break; }
        const std::string& sole_ibn = cur_node->op().input_bns().Get(0);
        if (!cur_node->SbpParallel4BnInOp(sole_ibn).has_broadcast_parallel()) { break; }
      }
      if (cur_node->op().output_bns().size() != 1) { break; }
      const std::string& sole_obn = cur_node->op().output_bns().Get(0);
      if (!cur_node->SbpParallel4BnInOp(sole_obn).has_broadcast_parallel()) { break; }
      if (cur_node->BatchAxis4Lbi(cur_node->op().BnInOp2Lbi(sole_obn)).has_value()) { break; }
      op_seq_without_batch_dim.push_back(cur_node);
      if (cur_node->out_edges().size() == 1) {
        cur_node = cur_node->SoleOutEdge()->dst_node();
      } else {
        cur_node = nullptr;
      }
    }
    if (op_seq_without_batch_dim.empty()) { return; }
    const OpNode* last_node = op_seq_without_batch_dim.back();
    const ParallelDesc& pd = last_node->parallel_desc();
    pd2last_node2node_seqs[pd][last_node] = op_seq_without_batch_dim;
    int64_t min_consumer_topo_order = node_cnt;
    last_node->ForEachNodeOnOutEdge([&](const OpNode* dst) {
      op_node2op_conf.emplace(dst, dst->op().op_conf());
      min_consumer_topo_order = std::min(min_consumer_topo_order, op_node2topo_order.at(dst));
    });
    last_node2model_size[last_node] = GetSoleOutBlobSize(node);
    last_node2order[last_node] = min_consumer_topo_order;
  });
  for (const auto& pair : pd2last_node2node_seqs) {
    const ParallelDesc& pd = pair.first;
    if (pd.parallel_num() <= 1) { continue; }
    std::vector<int64_t> parallel_id2size(pd.parallel_num(), 0);

    std::vector<std::pair<const OpNode*, int64_t>> last_node_out_size_pairs;
    const auto& last_node2node_seqs = pair.second;
    for (const auto& last_node7node_seqs : last_node2node_seqs) {
      const OpNode* last_node = last_node7node_seqs.first;
      last_node_out_size_pairs.emplace_back(last_node, last_node2model_size.at(last_node));
    }
    std::sort(
        last_node_out_size_pairs.begin(), last_node_out_size_pairs.end(),
        [&](const std::pair<const OpNode*, int64_t>& lhs,
            const std::pair<const OpNode*, int64_t>& rhs) { return lhs.second > rhs.second; });
    for (const auto& last_node7out_size : last_node_out_size_pairs) {
      const auto it = std::min_element(parallel_id2size.cbegin(), parallel_id2size.cend());
      const int64_t min_parallel_id = std::distance(parallel_id2size.cbegin(), it);
      last_node2parallel_id[last_node7out_size.first] = min_parallel_id;
      parallel_id2size[min_parallel_id] += last_node7out_size.second;
    }
    for (const auto& last_node7node_seqs : last_node2node_seqs) {
      const OpNode* last_node = last_node7node_seqs.first;
      const int64_t& parallel_id = last_node2parallel_id.at(last_node);
      for (const OpNode* node : last_node7node_seqs.second) {
        const ParallelConf parallel_conf = NonDistributedParallelConf4ParallelId(pd, parallel_id);
        builder->MutParallelConfOnlyOnce(node->op().op_name(), parallel_conf);
      }
      pd2last_nodes[pd].push_back(last_node);
    }
  }
  const int64_t group_size =
      GlobalJobDesc().non_distributed_optimizer_group_size_mbyte() * 1024 * 1024;
  CHECK_GE(group_size, 0);
  const int64_t half_group_size = group_size / 2;
  for (auto& pair : pd2last_nodes) {
    const ParallelDesc& pd = pair.first;
    std::vector<const OpNode*>* last_nodes = &pair.second;
    if (pd.parallel_num() <= 1) { continue; }
    std::sort(last_nodes->begin(), last_nodes->end(), [&](const OpNode* lhs, const OpNode* rhs) {
      return last_node2order.at(lhs) < last_node2order.at(rhs);
    });
    std::vector<std::vector<const OpNode*>> groups;
    int64_t cur_group_size = 0;
    for (const OpNode* node : *last_nodes) {
      const int64_t node_size = last_node2model_size.at(node);
      if (groups.empty() || cur_group_size > group_size
          || (cur_group_size >= half_group_size && node_size > group_size)) {
        groups.push_back({node});
        cur_group_size = node_size;
      } else {
        groups.back().push_back(node);
        cur_group_size += node_size;
      }
    }
    for (std::vector<const OpNode*>& group : groups) {
      std::sort(group.begin(), group.end(), [&](const OpNode* lhs, const OpNode* rhs) {
        return last_node2parallel_id.at(lhs) < last_node2parallel_id.at(rhs);
      });
      OperatorConf nccl_broadcast_op_conf{};
      nccl_broadcast_op_conf.set_name("System-Boxing-NcclTupleBroadcast-" + NewUniqueId());
      NcclTupleBroadcastOpConf* tuple_broadcast_conf =
          nccl_broadcast_op_conf.mutable_nccl_tuple_broadcast_conf();
      tuple_broadcast_conf->set_nccl_order_hint(op_node2topo_order.at(group.back()));
      FOR_RANGE(int64_t, i, 0, group.size()) {
        const OpNode* node = group.at(i);
        std::ostringstream ss;
        ss << "out_" << std::setw(6) << std::setfill('0') << i;
        const std::string obn = ss.str();
        const int64_t& parallel_id = last_node2parallel_id.at(node);
        const LogicalBlobId& lbi = node->op().BnInOp2Lbi(node->op().SoleObn());
        const BlobDesc& blob_desc = node->LogicalBlobDesc4Lbi(lbi);
        *tuple_broadcast_conf->mutable_in()->Add() = GenLogicalBlobName(lbi);
        *tuple_broadcast_conf->mutable_out()->Add() = obn;
        *tuple_broadcast_conf->mutable_root()->Add() = parallel_id;
        *tuple_broadcast_conf->mutable_data_type()->Add() = blob_desc.data_type();
        blob_desc.shape().ToProto(tuple_broadcast_conf->mutable_shape()->Add());
        const std::string new_lbn = nccl_broadcast_op_conf.name() + "/" + obn;
        node->ForEachNodeOnOutEdge([&](const OpNode* dst) {
          for (const std::string& ibn : dst->op().input_bns()) {
            if (dst->op().BnInOp2Lbi(ibn) == lbi) {
              PbMessage* dst_op_type_conf = MutableMessageInPbMessage(
                  &op_node2op_conf.at(dst), op_node2op_conf.at(dst).op_type_case());
              ReplaceStrValInPbFdOrPbRpf(dst_op_type_conf, ibn, GenLogicalBlobName(lbi), new_lbn);
            }
          }
        });
      }
      builder->AddOrMutOpsOnlyOnce(pd.parallel_conf(), {nccl_broadcast_op_conf});
    }
  }
  for (const auto& op_node7op_conf : op_node2op_conf) {
    builder->MutOpsOnlyOnce({op_node7op_conf.second});
  }
}

}  // namespace oneflow
