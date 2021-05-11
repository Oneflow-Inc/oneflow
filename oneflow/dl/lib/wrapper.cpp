#include "oneflow/dl/include/wrapper.h"
#include <dlfcn.h>

namespace oneflow {

namespace dl {

static void* checkDL(void* x) {
  if (!x) { LOG(ERROR) << "Error in dlopen or dlsym: " << dlerror(); }

  return x;
}
DynamicLibrary::DynamicLibrary(const char* name, const char* alt_name) {
  // NOLINTNEXTLINE(hicpp-signed-bitwise)
  handle_ = dlopen(name, RTLD_LOCAL | RTLD_NOW);
  if (!handle_) {
    if (alt_name) {
      handle_ = dlopen(alt_name, RTLD_LOCAL | RTLD_NOW);
      if (!handle_) { LOG(ERROR) << "Error in dlopen for library " << name << "and " << alt_name; }
    } else {
      LOG(ERROR) << "Error in dlopen: " << dlerror();
    }
  }
}

void* DynamicLibrary::sym(const char* name) {
  CHECK(handle_);
  return checkDL(dlsym(handle_, name));
}

DynamicLibrary::~DynamicLibrary() {
  if (!handle_) return;
  dlclose(handle_);
}

}  // namespace dl
}  // namespace oneflow
