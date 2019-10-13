#ifndef ONEFLOW_CORE_JOB_VERSION_H_
#define ONEFLOW_CORE_JOB_VERSION_H_

namespace oneflow {

#ifdef WITH_GIT_VERSION

const char* GetOneFlowGitVersion();

#endif  // WITH_GIT_VERSION

}

#endif //ONEFLOW_CORE_JOB_VERSION_H_
