#ifndef ONEFLOW_CORE_JOB_COMPLETER_ADD_PARALLEL_CAST_FACADE_OP_H_
#define ONEFLOW_CORE_JOB_COMPLETER_ADD_PARALLEL_CAST_FACADE_OP_H_

namespace oneflow {

class OpGraph;
class Job;

void AddParallelCastFacadeOp(const OpGraph &op_graph, Job *job);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_ADD_PARALLEL_CAST_FACADE_OP_H_
