#ifndef ONEFLOW_CORE_JOB_COMPLETER_AUTO_MIXED_PRECISION_H_
#define ONEFLOW_CORE_JOB_COMPLETER_AUTO_MIXED_PRECISION_H_

#include "oneflow/core/job_completer/auto_mixed_precision_lists.h"
#include "oneflow/core/job_completer/op_graph_pass.h"

namespace oneflow {

class OpGraph;
class OpNode;
class Job;

class AutoMixedPrecision final : public OpGraphPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AutoMixedPrecision);
  AutoMixedPrecision()
      : white_list_(AutoMixedPrecisionLists::WhiteList()),
        black_list_(AutoMixedPrecisionLists::BlackList()),
        gray_list_(AutoMixedPrecisionLists::GrayList()),
        clear_list_(AutoMixedPrecisionLists::ClearList()) {}
  ~AutoMixedPrecision() = default;

  bool IsEnabled() const override { return GlobalJobDesc().enable_auto_mixed_precision(); }
  void Apply(const OpGraph& op_graph, JobBuilder* job_builder) override;

 private:
  void FillBlackSet(const OpGraph& op_graph, HashSet<OpNode*>* black_set);
  void FillWhiteSet(const OpGraph& op_graph, std::function<bool(OpNode*)> IsAllowedToRunWithHalf,
                    const HashSet<OpNode*>& black_set, HashSet<OpNode*>* white_set);
  void PropagateWhiteThroughClearNodes(const OpGraph& op_graph,
                                       std::function<bool(OpNode*)> IsAllowedToRunWithHalf,
                                       const HashSet<OpNode*>& black_set,
                                       HashSet<OpNode*>* white_set);
  void InsertCastOp(const OpGraph& op_graph, const HashSet<OpNode*>& white_set,
                    JobBuilder* job_builder);

  const AMPList& white_list_;
  const AMPList& black_list_;
  const AMPList& gray_list_;
  const AMPList& clear_list_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_AUTO_MIXED_PRECISION_H_
