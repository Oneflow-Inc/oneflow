#ifndef ONEFLOW_CORE_JOB_COMPLETER_AGGREGATE_BOXING_H_
#define ONEFLOW_CORE_JOB_COMPLETER_AGGREGATE_BOXING_H_

namespace oneflow {

class OpGraph;
class Job;

void AggregateBoxingByDstParallel(const OpGraph &op_graph, Job *job);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_AGGREGATE_BOXING_H_
