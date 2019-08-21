#include "oneflow/core/job_completer/nccl_tuple_broadcast_group_pass.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

void NcclTupleBroadcastGroupPass::Apply(const OpGraph& op_graph, JobBuilder* builder) {
  HashMap<ParallelDesc, std::vector<const OpNode*>> pd2broadcast_nodes;
  HashMap<const OpNode*, OperatorConf> op_node2op_conf;
  HashMap<const OpNode*, int64_t> broadcast_node2out_size;
  op_graph.DfsTopoForEachNodeSortByDistanceToSink(
      op_graph.source_nodes(), &OpNode::ForEachNodeOnInEdge, &OpNode::ForEachNodeOnOutEdge,
      [&](const OpNode* node) {
        if (!node->op().op_conf().has_nccl_tuple_broadcast_conf()) { return; }
        pd2broadcast_nodes[node->parallel_desc()].push_back(node);
        node->ForEachNodeOnOutEdge(
            [&](const OpNode* dst) { op_node2op_conf[dst] = dst->op().op_conf(); });
        int64_t out_size = 0;
        for (const std::string& obn : node->op().output_bns()) {
          const BlobDesc& logical_blob_desc = node->LogicalBlobDesc4Lbi(node->op().BnInOp2Lbi(obn));
          out_size += logical_blob_desc.shape().elem_cnt()
                      * GetSizeOfDataType(logical_blob_desc.data_type());
        }
        broadcast_node2out_size[node] = out_size;
      });
  for (auto& pd7broadcast_nodes : pd2broadcast_nodes) {
    std::reverse(pd7broadcast_nodes.second.begin(), pd7broadcast_nodes.second.end());
  }
  for (const auto& pd7broadcast_nodes : pd2broadcast_nodes) {
    OperatorConf group_op_conf{};
    group_op_conf.set_name("System-Boxing-NcclTupleBroadcast-" + NewUniqueId());
    NcclTupleBroadcastOpConf* conf = group_op_conf.mutable_nccl_tuple_broadcast_conf();
    int64_t new_out_id = 0;
    for (const OpNode* node : pd7broadcast_nodes.second) {
      HashMap<LogicalBlobId, std::string> old_lbi2new_lbn;
      const NcclTupleBroadcastOpConf& old_conf = node->op().op_conf().nccl_tuple_broadcast_conf();
      FOR_RANGE(int64_t, i, 0, old_conf.in_size()) {
        *conf->mutable_in()->Add() = old_conf.in(i);
        *conf->mutable_root()->Add() = old_conf.root(i);
        *conf->mutable_shape()->Add() = old_conf.shape(i);
        *conf->mutable_data_type()->Add() = old_conf.data_type(i);
        const std::string new_out_bn = GenRepeatedBn("out", new_out_id);
        new_out_id += 1;
        *conf->mutable_out()->Add() = new_out_bn;
        old_lbi2new_lbn[GenLogicalBlobId(node->op().op_name() + "/" + old_conf.out(i))] =
            group_op_conf.name() + "/" + new_out_bn;
      }
      node->ForEachNodeOnOutEdge([&](const OpNode* dst) {
        for (const std::string& ibn : dst->op().input_bns()) {
          const LogicalBlobId& lbi = dst->op().BnInOp2Lbi(ibn);
          auto it = old_lbi2new_lbn.find(lbi);
          if (it != old_lbi2new_lbn.cend()) {
            PbMessage* dst_op_type_conf = MutableMessageInPbMessage(
                &op_node2op_conf.at(dst), op_node2op_conf.at(dst).op_type_case());
            SetBnValInOpTypeConf(dst_op_type_conf, ibn, GenLogicalBlobName(lbi),
                                 old_lbi2new_lbn.at(lbi));
          }
        }
      });
    }
    builder->AddOrMutOpsOnlyOnce(pd7broadcast_nodes.first.parallel_conf(), {group_op_conf});
  }
  for (const auto& op_node7op_conf : op_node2op_conf) {
    builder->MutOpsOnlyOnce({op_node7op_conf.second});
  }
  for (const auto& pd7broadcast_nodes : pd2broadcast_nodes) {
    for (const OpNode* node : pd7broadcast_nodes.second) {
      builder->DelOps({node->op().op_conf()});
    }
  }
}

}  // namespace oneflow
