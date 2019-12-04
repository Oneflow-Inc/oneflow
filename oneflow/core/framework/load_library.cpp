#include "oneflow/core/framework/load_library.h"

#include <dlfcn.h>

namespace oneflow {

Maybe<void> LoadLibrary(const std::string& lib_path) {
  void* handle = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  CHECK_OR_RETURN(handle) << " LoadLibrary ERROR! Cannot load library file: " + lib_path
    << " the Error is: " << dlerror();
  return Maybe<void>::Ok();
}

}  // namespace oneflow
