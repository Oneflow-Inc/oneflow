#ifndef ONEFLOW_CORE_JOB_COMPLETER_AUTO_MIXED_PRECISION_H_
#define ONEFLOW_CORE_JOB_COMPLETER_AUTO_MIXED_PRECISION_H_

#include "oneflow/core/job_completer/auto_mixed_precision_lists.h"

namespace oneflow {

class OpGraph;
class OpNode;
class Job;

class AutoMixedPrecision final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AutoMixedPrecision);
  AutoMixedPrecision(const AMPList& white, const AMPList& black, const AMPList& gray)
      : white_list_(white), black_list_(black), gray_list_(gray) {}
  ~AutoMixedPrecision() = default;

  void Apply(const OpGraph& op_graph, Job* job);

 private:
  void FillBlackSet(const OpGraph& op_graph, HashSet<OpNode*>* black_set);
  void FillWhiteSet(const OpGraph& op_graph, std::function<bool(OpNode*)> IsAllowedToRunWithHalf,
                    const HashSet<OpNode*>& black_set, HashSet<OpNode*>* white_set);
  void PropagateWhiteThroughNonListNodes(const OpGraph& op_graph,
                                         std::function<bool(OpNode*)> IsAllowedToRunWithHalf,
                                         const HashSet<OpNode*>& black_set,
                                         HashSet<OpNode*>* white_set);
  void InsertCastOp(const OpGraph& op_graph, const HashSet<OpNode*>& white_set, Job* job);

  const AMPList& white_list_;
  const AMPList& black_list_;
  const AMPList& gray_list_;
  HashSet<OpNode*> non_list_nodes_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_AUTO_MIXED_PRECISION_H_
