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
#ifndef ONEFLOW_CORE_FRAMEWORK_INFER_OUTPUT_BLOB_TIME_SHAPE_FN_CONTEXT_H_
#define ONEFLOW_CORE_FRAMEWORK_INFER_OUTPUT_BLOB_TIME_SHAPE_FN_CONTEXT_H_

#include "oneflow/core/framework/user_op_conf.h"

namespace oneflow {

namespace user_op {

class InferOutputBlobTimeShapeFnContext {
 public:
  InferOutputBlobTimeShapeFnContext() = default;
  virtual ~InferOutputBlobTimeShapeFnContext() = default;
  InferOutputBlobTimeShapeFnContext(const InferOutputBlobTimeShapeFnContext&) = delete;

  virtual const Shape& TimeShape4InputArgNameAndIndex(const std::string& arg_name,
                                                      int32_t index) = 0;
  virtual const UserOpConfWrapper& user_op_conf() const = 0;
  virtual Shape* mut_output_blob_time_shape() = 0;
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_INFER_OUTPUT_BLOB_TIME_SHAPE_FN_CONTEXT_H_
