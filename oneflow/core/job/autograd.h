#ifndef ONEFLOW_CORE_JOB_AUTOGRAD_H_
#define ONEFLOW_CORE_JOB_AUTOGRAD_H_

#include "oneflow/core/job/job_desc.h"

namespace oneflow {

JobConf1 AutoGrad(const JobDesc& job_desc);
}

#endif  // ONEFLOW_CORE_JOB_AUTOGRAD_H_
