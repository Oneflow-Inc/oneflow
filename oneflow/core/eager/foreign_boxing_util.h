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
#ifndef ONEFLOW_CORE_FRAMEWORK_FOREIGN_BOXING_UTIL_H_
#define ONEFLOW_CORE_FRAMEWORK_FOREIGN_BOXING_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/op_arg_util.h"

namespace oneflow {

class ForeignBoxingUtil {
 public:
  ForeignBoxingUtil() = default;
  virtual ~ForeignBoxingUtil() = default;

  virtual std::shared_ptr<compatible_py::BlobObject> BoxingTo(
      const std::shared_ptr<InstructionsBuilder>& builder,
      const std::shared_ptr<compatible_py::BlobObject>& blob_object,
      const std::shared_ptr<compatible_py::OpArgParallelAttribute>& op_arg_parallel_attr) const {
    UNIMPLEMENTED();
  }

  virtual std::shared_ptr<ParallelDesc> TryReplaceDeviceTag(
      const std::shared_ptr<InstructionsBuilder>& builder,
      const std::shared_ptr<ParallelDesc>& parallel_desc_symbol,
      const std::string& device_tag) const {
    UNIMPLEMENTED();
  }

  virtual void Assign(const std::shared_ptr<InstructionsBuilder>& builder,
                      const std::shared_ptr<compatible_py::BlobObject>& target_blob_object,
                      const std::shared_ptr<compatible_py::BlobObject>& source_blob_object) const {
    UNIMPLEMENTED();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_FOREIGN_BOXING_UTIL_H_
