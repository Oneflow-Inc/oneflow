#ifndef ONEFLOW_CORE_JOB_COMPLETER_AUTO_MIXED_PRECISION_H_
#define ONEFLOW_CORE_JOB_COMPLETER_AUTO_MIXED_PRECISION_H_

#include "oneflow/core/job_completer/auto_mixed_precision_lists.h"
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

class OpGraph;
class OpNode;
class Job;

class AutoMixedPrecision final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AutoMixedPrecision);
  AutoMixedPrecision(const AMPList& white, const AMPList& black, const AMPList& gray,
                     const AMPList& clear)
      : white_list_(white), black_list_(black), gray_list_(gray), clear_list_(clear) {}
  ~AutoMixedPrecision() = default;

  void Apply(const OpGraph& op_graph, JobBuilder* job_builder);

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
