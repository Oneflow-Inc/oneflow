#include "oneflow/core/job_completer/nccl_tuple_broadcast_reduce_sequence_pass.h"
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

void NcclTupleBroadcastReduceSequencePass::Apply(const OpGraph& op_graph, JobBuilder* builder) {
  std::vector<OperatorConf> broadcast_ops;
  std::vector<OperatorConf> reduce_ops;
  op_graph.ForEachNode([&](const OpNode* node) {
    const OperatorConf& op_conf = node->op().op_conf();
    if (op_conf.has_nccl_tuple_broadcast_conf()) {
      broadcast_ops.push_back(op_conf);
    } else if (op_conf.has_nccl_tuple_reduce_conf()) {
      reduce_ops.push_back(op_conf);
    }
  });
  std::sort(broadcast_ops.begin(), broadcast_ops.end(),
            [](const OperatorConf& lhs, const OperatorConf& rhs) {
              return lhs.nccl_tuple_broadcast_conf().nccl_order_hint()
                     < rhs.nccl_tuple_broadcast_conf().nccl_order_hint();
            });
  std::sort(reduce_ops.begin(), reduce_ops.end(),
            [](const OperatorConf& lhs, const OperatorConf& rhs) {
              return lhs.nccl_tuple_reduce_conf().nccl_order_hint()
                     < rhs.nccl_tuple_reduce_conf().nccl_order_hint();
            });
  FOR_RANGE(int64_t, i, 1, broadcast_ops.size()) {
    broadcast_ops.at(i).add_ctrl_in_op_name(broadcast_ops.at(i - 1).name());
  }
  FOR_RANGE(int64_t, i, 1, reduce_ops.size()) {
    reduce_ops.at(i).add_ctrl_in_op_name(reduce_ops.at(i - 1).name());
  }
  if (!broadcast_ops.empty() && !reduce_ops.empty()) {
    reduce_ops.front().add_ctrl_in_op_name(broadcast_ops.back().name());
    if (broadcast_ops.size() + reduce_ops.size() > 2) {
      reduce_ops.back().add_ctrl_in_op_name(broadcast_ops.front().name());
    }
  }
  builder->MutOpsOnlyOnce(broadcast_ops);
  builder->MutOpsOnlyOnce(reduce_ops);
}

}  // namespace oneflow
