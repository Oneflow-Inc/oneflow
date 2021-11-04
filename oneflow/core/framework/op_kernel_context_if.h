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
#ifndef ONEFLOW_CORE_FRAMEWORK_OP_KERNEL_CONTEXT_IF_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_KERNEL_CONTEXT_IF_H_
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/user_op_tensor.h"
#include "oneflow/core/framework/attr_value.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/stream/include/stream_context.h"
#include "oneflow/core/job/sbp_parallel.cfg.h"
#include "oneflow/core/framework/to_string.h"

namespace oneflow {
namespace user_op {

class OpInfoIf {
 public:
  virtual const std::string& input(const std::string& arg_name, int32_t index) const = 0;
  virtual const std::string& output(const std::string& arg_name, int32_t index) const = 0;
  virtual bool has_input(const std::string& arg_name, int32_t index) const = 0;
  virtual bool has_output(const std::string& arg_name, int32_t index) const = 0;
  virtual int32_t input_size(const std::string& arg_name) const = 0;
  virtual int32_t output_size(const std::string& arg_name) const = 0;
  virtual const std::string& op_name() const = 0;
  virtual const std::string& op_type_name() const = 0;
};

class DeviceInfoIf {
 public:
  virtual const std::string& device_tag() const = 0;
  virtual DeviceType device_type() const { return CHECK_JUST(DeviceType4DeviceTag(device_tag())); }
};

class AttrIf {
 public:
  template<typename T>
  const T& Attr(const std::string& attr_name) const {
    return AttrValueCast<T>(*Attr4Name(attr_name));
  }

 protected:
  virtual const std::shared_ptr<const AttrVal>& Attr4Name(const std::string& attr_name) const = 0;
};

class UserOpConfAttrProvider : virtual public AttrIf {
 public:
  explicit UserOpConfAttrProvider(const UserOpConfWrapper& user_op_conf)
      : user_op_conf_(user_op_conf) {}

 protected:
  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override {
    return user_op_conf_.Attr4Name(attr_name);
  }

 private:
  UserOpConfWrapper user_op_conf_;
};

class UserOpConfOpInfoProvider : virtual public OpInfoIf, virtual public DeviceInfoIf {
 public:
  explicit UserOpConfOpInfoProvider(const UserOpConfWrapper& user_op_conf)
      : user_op_conf_(user_op_conf) {}
  const std::string& input(const std::string& arg_name, int32_t index) const override {
    return user_op_conf().input(arg_name, index);
  }
  const std::string& output(const std::string& arg_name, int32_t index) const override {
    return user_op_conf().output(arg_name, index);
  }
  bool has_input(const std::string& arg_name, int32_t index) const override {
    return user_op_conf().has_input(arg_name, index);
  }
  bool has_output(const std::string& arg_name, int32_t index) const override {
    return user_op_conf().has_output(arg_name, index);
  }
  int32_t input_size(const std::string& arg_name) const override {
    return user_op_conf().input_size(arg_name);
  }
  int32_t output_size(const std::string& arg_name) const override {
    return user_op_conf().output_size(arg_name);
  }
  const std::string& op_name() const override { return user_op_conf().op_name(); }
  const std::string& op_type_name() const override { return user_op_conf().op_type_name(); }
  const std::string& device_tag() const override { return user_op_conf().op_conf().device_tag(); }

 protected:
  const UserOpConfWrapper& user_op_conf() const { return user_op_conf_; }

 private:
  UserOpConfWrapper user_op_conf_;
};

class InputAndOutputNameIf {
 public:
  virtual const std::vector<std::pair<std::string, int32_t>>& inputs() const = 0;
  virtual const std::vector<std::pair<std::string, int32_t>>& outputs() const = 0;
};

class StreamCtxAndDeviceCtxIf {
 public:
  virtual DeviceCtx* device_ctx() = 0;
  virtual StreamContext* stream_ctx() = 0;
};

class ConsistentInfoIf {
 public:
  virtual const cfg::SbpParallel& SbpParallel4ArgNameAndIndex(const std::string& arg_name,
                                                              int32_t index) const {
    const auto& nd_sbp = NdSbp4ArgNameAndIndex(arg_name, index);
    CHECK(nd_sbp.sbp_parallel_size() == 1);
    return nd_sbp.sbp_parallel(0);
  }
  virtual const TensorDesc* LogicalTensorDesc4ArgNameAndIndex(const std::string&,
                                                              int32_t) const = 0;
  virtual const cfg::NdSbp& NdSbp4ArgNameAndIndex(const std::string&, int32_t) const = 0;
};

class TensorDescIf {
 public:
  virtual const TensorDesc* TensorDesc4ArgNameAndIndex(const std::string&, int32_t) const = 0;
};

class TensorObjIf {
 public:
  virtual Tensor* Tensor4ArgNameAndIndex(const std::string& arg_name, int32_t arg_index) = 0;
};

}  // namespace user_op
}  // namespace oneflow
#endif  // ONEFLOW_CORE_FRAMEWORK_OP_KERNEL_CONTEXT_IF_H_
