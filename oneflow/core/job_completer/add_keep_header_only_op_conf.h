#ifndef ONEFLOW_CORE_JOB_COMPLETER_ADD_KEEP_HEADER_ONLY_H_
#define ONEFLOW_CORE_JOB_COMPLETER_ADD_KEEP_HEADER_ONLY_H_

namespace oneflow {

class OpGraph;
class Job;

void AddKeepHeaderOnlyOp(const OpGraph& op_graph, Job* job);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_ADD_KEEP_HEADER_ONLY_H_
