#ifndef ONEFLOW_CORE_AUTOGRAD_AUTO_SAVER_H_
#define ONEFLOW_CORE_AUTOGRAD_AUTO_SAVER_H_

#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

void AutoSaver(const OpGraph& op_graph, JobConf1* job_conf);

}

#endif  // ONEFLOW_CORE_AUTOGRAD_AUTO_SAVER_H_
