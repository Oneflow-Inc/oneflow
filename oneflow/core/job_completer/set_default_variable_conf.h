#ifndef ONEFLOW_CORE_JOB_COMPLETER_FILL_VARIABLE_CONF_H_
#define ONEFLOW_CORE_JOB_COMPLETER_FILL_VARIABLE_CONF_H_

#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

void SetDefaultVariableConf(const OpGraph& op_graph, JobBuilder* job_builder);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_FILL_VARIABLE_CONF_H_
