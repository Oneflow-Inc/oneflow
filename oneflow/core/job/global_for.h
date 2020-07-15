#ifndef ONEFLOW_CORE_JOB_GLOBAL_FOR_H_
#define ONEFLOW_CORE_JOB_GLOBAL_FOR_H_

#include "oneflow/core/common/global.h"

namespace oneflow {

class ForSession {};
class ForEnv {};

template<typename T>
class EagerExecution {};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_GLOBAL_FOR_H_
