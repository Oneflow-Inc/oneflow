#ifndef ONEFLOW_CORE_JOB_COMPLETER_GROUP_BOXING_BY_DST_PARALLEL_H_
#define ONEFLOW_CORE_JOB_COMPLETER_GROUP_BOXING_BY_DST_PARALLEL_H_

namespace oneflow {

class OpGraph;
class Job;

void GroupBoxingByDstParallel(const OpGraph &op_graph, Job *job);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_GROUP_BOXING_BY_DST_PARALLEL_H_
