#ifndef ONEFLOW_CORE_JOB_COMPLETER_AUTO_LEARNING_RATE_H_
#define ONEFLOW_CORE_JOB_COMPLETER_AUTO_LEARNING_RATE_H_

namespace oneflow {

class OpGraph;
class Job;

void AutoLearningRate(const OpGraph& op_graph, Job* job);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_AUTO_LEARNING_RATE_H_
