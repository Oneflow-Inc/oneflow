#ifndef ONEFLOW_CORE_JOB_COMPLETER_AUTO_GLOBAL_STEP_H_
#define ONEFLOW_CORE_JOB_COMPLETER_AUTO_GLOBAL_STEP_H_

namespace oneflow {

class OpGraph;
class Job;

void AutoGlobalStep(const OpGraph& op_graph, Job* job);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_AUTO_GLOBAL_STEP_H_
