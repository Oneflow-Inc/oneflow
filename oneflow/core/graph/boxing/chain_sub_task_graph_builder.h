#ifndef ONEFLOW_CORE_GRAPH_BOXING_CHAIN_SUB_TASK_GRAPH_BUILDER_H_
#define ONEFLOW_CORE_GRAPH_BOXING_CHAIN_SUB_TASK_GRAPH_BUILDER_H_

#include "oneflow/core/graph/boxing/sub_task_graph_builder.h"

namespace oneflow {

class ChainSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChainSubTskGphBuilder);
  explicit ChainSubTskGphBuilder(std::vector<std::shared_ptr<SubTskGphBuilder>> builders)
      : builders_(std::move(builders)) {}
  ~ChainSubTskGphBuilder() override = default;

  Maybe<void> Build(SubTskGphBuilderCtx* ctx,
                    const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
                    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,
                    const ParallelDesc& src_parallel_desc, const ParallelDesc& dst_parallel_desc,
                    const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc,
                    const SbpParallel& src_sbp_parallel,
                    const SbpParallel& dst_sbp_parallel) const override;

 private:
  std::vector<std::shared_ptr<SubTskGphBuilder>> builders_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BOXING_CHAIN_SUB_TASK_GRAPH_BUILDER_H_
