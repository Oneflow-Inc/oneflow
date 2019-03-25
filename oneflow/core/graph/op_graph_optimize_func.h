#ifndef ONEFLOW_CORE_GRAPH_OP_GRAPH_OPTIMIZE_FUNC_H_
#define ONEFLOW_CORE_GRAPH_OP_GRAPH_OPTIMIZE_FUNC_H_

namespace oneflow {

class OpGraph;
class JobConf1;

void AddKeepHeaderOnlyOp(const OpGraph& op_graph, JobConf1* job_conf);

} // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_OP_GRAPH_OPTIMIZE_FUNC_H_
