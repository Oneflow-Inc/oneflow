#ifndef ONEFLOW_CORE_JOB_COMPLETER_ADD_SAVER_H_
#define ONEFLOW_CORE_JOB_COMPLETER_ADD_SAVER_H_

#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

void AddSaver(const OpGraph &op_graph, Job *job_conf);
}

#endif  // ONEFLOW_CORE_JOB_COMPLETER_ADD_SAVER_H_
