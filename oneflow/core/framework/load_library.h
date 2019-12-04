#ifndef ONEFLOW_CORE_FRAMEWORK_LOAD_LIBRARY_H_
#define ONEFLOW_CORE_FRAMEWORK_LOAD_LIBRARY_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {

Maybe<void> LoadLibrary(const std::string& lib_path);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_LOAD_LIBRARY_H_
