#include "oneflow/core/job/version.h"

namespace oneflow {

void DumpVersionInfo() {
#ifdef WITH_GIT_VERSION
  LOG(INFO) << "OneFLow Version: " << GetOneFlowGitVersion();
#endif  // WITH_GIT_VERSION
}

}  // namespace oneflow
