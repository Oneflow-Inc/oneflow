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
#ifndef ONEFLOW_CORE_FRAMEWORK_INFER_UTIL_H_
#define ONEFLOW_CORE_FRAMEWORK_INFER_UTIL_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/tensor_desc.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/job/sbp_parallel.pb.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {

class Shape;
class JobDesc;

namespace user_op {

class UserOpDefWrapper;

class InferContext {
 public:
  virtual ~InferContext() = default;

  virtual TensorDesc* TensorDesc4ArgNameAndIndex(const std::string&, int32_t) = 0;
  virtual const TensorDesc* LogicalTensorDesc4ArgNameAndIndex(const std::string&,
                                                              int32_t) const = 0;
  virtual Shape* Shape4ArgNameAndIndex(const std::string&, int32_t) = 0;
  virtual DataType* Dtype4ArgNameAndIndex(const std::string&, int32_t) = 0;
  virtual const std::vector<std::pair<std::string, int32_t>>& inputs() const = 0;
  virtual const std::vector<std::pair<std::string, int32_t>>& outputs() const = 0;
  const std::string& input(const std::string& arg_name, int32_t index) const {
    return user_op_conf().input(arg_name, index);
  }
  const std::string& output(const std::string& arg_name, int32_t index) const {
    return user_op_conf().output(arg_name, index);
  }
  bool has_input(const std::string& arg_name, int32_t index) const {
    return user_op_conf().has_input(arg_name, index);
  }
  bool has_output(const std::string& arg_name, int32_t index) const {
    return user_op_conf().has_output(arg_name, index);
  }
  int32_t input_size(const std::string& arg_name) const {
    return user_op_conf().input_size(arg_name);
  }
  int32_t output_size(const std::string& arg_name) const {
    return user_op_conf().output_size(arg_name);
  }
  const std::string& op_name() const { return user_op_conf().op_name(); }
  const std::string& op_type_name() const { return user_op_conf().op_type_name(); }
  const std::string& device_tag() const { return user_op_conf().op_conf().device_tag(); }

  bool has_attr(const std::string& attr_name) const {
    const auto& attr_name2attr_val = user_op_conf().attrs();
    return attr_name2attr_val.find(attr_name) != attr_name2attr_val.end();
  }
  template<typename T>
  const T& Attr(const std::string& attr_name) const;

  virtual const ParallelContext& parallel_ctx() const = 0;
  virtual const ParallelDesc& parallel_desc() const = 0;

  virtual const JobDesc* job_desc() const {
    UNIMPLEMENTED();
    return nullptr;
  };
  virtual const SbpParallel& SbpParallel4ArgNameAndIndex(const std::string&, int32_t) const = 0;

  virtual const ParallelDistribution& ParallelDistribution4ArgNameAndIndex(const std::string&,
                                                                           int32_t) const = 0;

  virtual bool* IsDynamic4ArgNameAndIndex(const std::string&, int32_t) = 0;

  virtual int64_t parallel_num() const = 0;

 protected:
  InferContext() = default;
  InferContext(const InferContext&) = delete;
  virtual const UserOpConfWrapper& user_op_conf() const = 0;
  virtual const std::shared_ptr<AttrVal>& Attr4AttrName(const std::string& attr_name) const = 0;
};

class DeviceInferContext {
 public
  virtual ~DeviceInferContext() = default;

  virtual const std::vector<std::pair<std::string, int32_t>>& inputs() const = 0;
  virtual const std::vector<std::pair<std::string, int32_t>>& outputs() const = 0;

  virtual std::shared_ptr<const Device>* OutputTensorDevice4ArgNameAndIndex(const std::string&, int32_t) = 0;

  virtual const std::shared_ptr<const Device>& InputTensorDevice4ArgNameAndIndex(const std::string&, int32_t) const = 0;

  template<typename T>
  const T& Attr(const std::string& attr_name) const;

  virtual bool HasAttr(const std::string& attr_name) const = 0;

 protected:
  DeviceInferContext() = default;
  virtual const std::shared_ptr<AttrVal>& Attr4AttrName(const std::string& attr_name) const = 0;
};

struct TensorDescInferFnUtil {
  static Maybe<void> Unchanged(InferContext*);
  static Maybe<void> UnchangedDataType(InferContext*);
  static Maybe<void> InOutCorrespond(InferContext*);
};

struct CheckAttrFnUtil {
  static Maybe<void> NoCheck(const UserOpDefWrapper&, const UserOpConfWrapper&);
};

struct TmpSizeInferFnUtil {
  static size_t ZeroTmpSize(InferContext*);
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_INFER_UTIL_H_
