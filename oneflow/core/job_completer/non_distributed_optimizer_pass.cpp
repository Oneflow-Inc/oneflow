#include "oneflow/core/job_completer/non_distributed_optimizer_pass.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

namespace {

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
  parallel_conf.set_policy(pd.policy());
  return parallel_conf;
}

}  // namespace

void NonDistributedOptimizerPass::Apply(const OpGraph& op_graph, Job* job) {
  JobBuilder builder(job);
  HashMap<ParallelDesc, HashMap<const OpNode*, std::vector<const OpNode*>>> pd2last_node2node_seqs;
  HashMap<const OpNode*, OperatorConf> op_node2op_conf;
  HashMap<const OpNode*, int64_t> last_node2out_size;
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
      if (cur_node->HasBatchDim4Lbi(cur_node->op().BnInOp2Lbi(sole_obn))) { break; }
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
    last_node->ForEachNodeOnOutEdge(
        [&](const OpNode* dst) { op_node2op_conf.emplace(dst, dst->op().op_conf()); });
    const BlobDesc& last_node_out_logical_blob_desc = last_node->LogicalBlobDesc4Lbi(
        last_node->op().BnInOp2Lbi(last_node->op().output_bns().Get(0)));
    last_node2out_size[last_node] =
        last_node_out_logical_blob_desc.shape().elem_cnt()
        * GetSizeOfDataType(last_node_out_logical_blob_desc.data_type());
  });
  for (const auto& pair : pd2last_node2node_seqs) {
    const ParallelDesc& pd = pair.first;
    if (pd.parallel_num() <= 1) { continue; }
    std::vector<int64_t> parallel_id2size(pd.parallel_num(), 0);
    HashMap<const OpNode*, int64_t> last_node2parallel_id;
    std::vector<std::pair<const OpNode*, int64_t>> last_node_out_size_pairs;
    const auto& last_node2node_seqs = pair.second;
    for (const auto& last_node7node_seqs : last_node2node_seqs) {
      const OpNode* last_node = last_node7node_seqs.first;
      last_node_out_size_pairs.emplace_back(last_node, last_node2out_size.at(last_node));
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
        builder.MutOpsWithNewParallelConf(parallel_conf, {node->op().op_conf()});
      }
      OperatorConf nccl_broadcast_op_conf{};
      nccl_broadcast_op_conf.set_name("System-Boxing-NcclTupleBroadcast-" + NewUniqueId());
      NcclTupleBroadcastOpConf* tuple_broadcast_conf =
          nccl_broadcast_op_conf.mutable_nccl_tuple_broadcast_conf();
      const LogicalBlobId& lbi = last_node->op().BnInOp2Lbi(last_node->op().SoleObn());
      const BlobDesc& blob_desc = last_node->LogicalBlobDesc4Lbi(lbi);
      *tuple_broadcast_conf->mutable_in()->Add() = GenLogicalBlobName(lbi);
      *tuple_broadcast_conf->mutable_out()->Add() = "out";
      *tuple_broadcast_conf->mutable_root()->Add() = parallel_id;
      *tuple_broadcast_conf->mutable_data_type()->Add() = blob_desc.data_type();
      blob_desc.shape().ToProto(tuple_broadcast_conf->mutable_shape()->Add());
      builder.AddOrMutOps(pd.parallel_conf(), {nccl_broadcast_op_conf});
      const std::string new_lbn = nccl_broadcast_op_conf.name() + "/out";
      last_node->ForEachNodeOnOutEdge([&](const OpNode* dst) {
        for (const std::string& ibn : dst->op().input_bns()) {
          if (dst->op().BnInOp2Lbi(ibn) == lbi) {
            PbMessage* dst_op_type_conf = MutableMessageInPbMessage(
                &op_node2op_conf.at(dst), op_node2op_conf.at(dst).op_type_case());
            SetBnValInOpTypeConf(dst_op_type_conf, ibn, GenLogicalBlobName(lbi), new_lbn);
          }
        }
      });
    }
    for (const auto& op_node7op_conf : op_node2op_conf) {
      builder.MutOps({op_node7op_conf.second});
    }
  }
}

}  // namespace oneflow
