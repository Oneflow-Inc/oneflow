/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_COMMON_EXCEPTION_H_
#define ONEFLOW_CORE_COMMON_EXCEPTION_H_

#include <exception>
#include <string>
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

class Exception : public std::exception {
 public:
  Exception(const std::string& what) : what_(what) {}
  virtual ~Exception() = default;

  const char* what() const noexcept override { return what_.c_str(); }

 private:
  std::string what_;
};

#define EXCEPTION_SEQ                             \
  OF_PP_MAKE_TUPLE_SEQ(ConfigAssertFailed)        \
  OF_PP_MAKE_TUPLE_SEQ(ConfigResourceUnavailable) \
  OF_PP_MAKE_TUPLE_SEQ(ProtoParseFailed)          \
  OF_PP_MAKE_TUPLE_SEQ(CheckFailed)               \
  OF_PP_MAKE_TUPLE_SEQ(Todo)                      \
  OF_PP_MAKE_TUPLE_SEQ(Unimplemented)             \
  OF_PP_MAKE_TUPLE_SEQ(BoxingNotSupported)        \
  OF_PP_MAKE_TUPLE_SEQ(GradientFunctionNotFound)  \
  OF_PP_MAKE_TUPLE_SEQ(OpKernelNotFound)          \
  OF_PP_MAKE_TUPLE_SEQ(MultipleOpKernelsMatched)  \
  OF_PP_MAKE_TUPLE_SEQ(MemoryZoneOutOfMemory)     \
  OF_PP_MAKE_TUPLE_SEQ(LossBlobNotFound)          \
  OF_PP_MAKE_TUPLE_SEQ(JobSetEmpty)               \
  OF_PP_MAKE_TUPLE_SEQ(DeviceTagNotFound)         \
  OF_PP_MAKE_TUPLE_SEQ(JobNameExist)              \
  OF_PP_MAKE_TUPLE_SEQ(JobNameEmpty)              \
  OF_PP_MAKE_TUPLE_SEQ(JobNameNotEqual)           \
  OF_PP_MAKE_TUPLE_SEQ(NoJobBuildAndInferCtx)     \
  OF_PP_MAKE_TUPLE_SEQ(JobConfFrozen)             \
  OF_PP_MAKE_TUPLE_SEQ(JobConfNotSet)             \
  OF_PP_MAKE_TUPLE_SEQ(JobConfRepeatedSet)        \
  OF_PP_MAKE_TUPLE_SEQ(JobTypeNotSet)             \
  OF_PP_MAKE_TUPLE_SEQ(LogicalBlobNameNotExist)   \
  OF_PP_MAKE_TUPLE_SEQ(LogicalBlobNameExist)      \
  OF_PP_MAKE_TUPLE_SEQ(LogicalBlobNameInvalid)    \
  OF_PP_MAKE_TUPLE_SEQ(OpNameExist)               \
  OF_PP_MAKE_TUPLE_SEQ(OpConfDeviceTagNoSet)      \
  OF_PP_MAKE_TUPLE_SEQ(Placement)                 \
  OF_PP_MAKE_TUPLE_SEQ(BlobSplitAxisInfer)        \
  OF_PP_MAKE_TUPLE_SEQ(UnknownJobBuildAndInfer)   \
  OF_PP_MAKE_TUPLE_SEQ(RwMutexedObjectNotFound)   \
  OF_PP_MAKE_TUPLE_SEQ(SymbolIdUninitialized)     \
  OF_PP_MAKE_TUPLE_SEQ(Unknown)                   \
  OF_PP_MAKE_TUPLE_SEQ(CompileOptionWrong)

#define DEFINE_EXCEPTION_CLASS(cls)                                         \
  class OF_PP_CAT(cls, Exception) : public Exception {                      \
   public:                                                                  \
    OF_PP_CAT(cls, Exception)(const std::string& what) : Exception(what) {} \
    ~OF_PP_CAT(cls, Exception)() override = default;                        \
  };

OF_PP_FOR_EACH_TUPLE(DEFINE_EXCEPTION_CLASS, EXCEPTION_SEQ)

#undef DEFINE_EXCEPTION_CLASS

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_EXCEPTION_H_
