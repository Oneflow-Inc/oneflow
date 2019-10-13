#ifndef ONEFLOW_CORE_JOB_VERSION_H_
#define ONEFLOW_CORE_JOB_VERSION_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

#ifdef WITH_GIT_VERSION

const char* GetOneFlowGitVersion();

#endif  // WITH_GIT_VERSION

void DumpVersionInfo();

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_VERSION_H_
