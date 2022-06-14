#ifndef ONEFLOW_CORE_COMMON_ARRAY_REF_H_
#define ONEFLOW_CORE_COMMON_ARRAY_REF_H_

#include "llvm/ADT/ArrayRef.h"

namespace oneflow {

template<typename T>
using ArrayRef = llvm::ArrayRef<T>;

template<typename T>
using MutableArrayRef = llvm::MutableArrayRef<T>;

}

#endif
