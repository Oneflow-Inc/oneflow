
#include "OneFlow/Extension.h"
namespace oneflow {

SharedLibs* MutSharedLibPaths() {
  static SharedLibs libs = {};
  return &libs;
}

const SharedLibs* SharedLibPaths() { return MutSharedLibPaths(); }
}  // namespace oneflow
