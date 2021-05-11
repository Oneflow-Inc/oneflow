#ifndef ONEFLOW_CORE_DL_INCLUDE_WRAPPER_H_
#define ONEFLOW_CORE_DL_INCLUDE_WRAPPER_H_

#include "oneflow/core/common/util.h"
#if defined(WITH_RDMA)

#include <infiniband/verbs.h>

namespace oneflow {

namespace dl {
struct DynamicLibrary {
  OF_DISALLOW_COPY_AND_MOVE(DynamicLibrary);

  DynamicLibrary(const char* name, const char* alt_name = nullptr);

  void* sym(const char* name);

  ~DynamicLibrary();

 private:
  void* handle_ = nullptr;
};
}  // namespace dl

}  // namespace oneflow

#endif  // WITH_RDMA

#endif  // ONEFLOW_CORE_DL_INCLUDE_WRAPPER_H_
