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
#ifndef ONEFLOW_CORE_FRAMEWORK_USER_OP_CONF_H_
#define ONEFLOW_CORE_FRAMEWORK_USER_OP_CONF_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/tensor_desc.h"
#include "oneflow/core/framework/user_op_def.pb.h"
#include "oneflow/core/framework/user_op_attr.pb.h"
#include "oneflow/core/framework/user_op_conf.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

class BlobDesc;

namespace user_op {

class OpArg final {
 public:
  OpArg(std::string&& name, int32_t index) : name_(std::move(name)), index_(index) {}

  const std::string& name() const { return name_; }
  int32_t index() const { return index_; }

 private:
  std::string name_;
  int32_t index_;
};

class AttrVal {
 public:
  AttrVal() = default;
  virtual ~AttrVal() = default;

 private:
  OF_DISALLOW_COPY_AND_MOVE(AttrVal)
};

template<typename T>
class TypedAttrVal final : public AttrVal {
 public:
  TypedAttrVal(T v) : val_(v) {}
  ~TypedAttrVal() = default;

  const T& val() const { return val_; }

 private:
  OF_DISALLOW_COPY_AND_MOVE(TypedAttrVal)

  T val_;
};

class UserOpConfWrapper final {
 public:
  UserOpConfWrapper(const OperatorConf&);
  const OperatorConf& op_conf() const;
  const UserOpConf& user_op_conf() const;
  const std::string& op_name() const;
  const std::string& op_type_name() const;
  const std::string& input(const std::string& arg_name, int32_t index) const;
  const std::string& output(const std::string& arg_name, int32_t index) const;
  bool has_input(const std::string& arg_name, int32_t index) const;
  bool has_output(const std::string& arg_name, int32_t index) const;
  int32_t input_size(const std::string& arg_name) const;
  int32_t output_size(const std::string& arg_name) const;

  template<typename T>
  const T& attr(const std::string& attr_name) const;

 private:
  UserOpConfWrapper() = default;
  friend class UserOpConfWrapperBuilder;

  OperatorConf op_conf_;
  HashMap<std::string, std::shared_ptr<AttrVal>> attrs_;
};

using UserOpInputGradGetFn = std::function<const std::string&()>;
class UserOpWrapper final {
 public:
  UserOpWrapper(const OperatorConf& op, const std::function<const BlobDesc&(const std::string&)>&,
                const std::function<LogicalBlobId*(const std::string&)>&);

 public:
  const UserOpConfWrapper& user_op_conf() const { return conf_; }
  const OperatorConf& op_conf() const { return conf_.op_conf(); }
  const std::string& op_name() const { return conf_.op_name(); }
  const std::string& op_type_name() const { return conf_.op_type_name(); }

  int32_t input_size(const std::string& arg_name) const { return conf_.input_size(arg_name); }
  const std::string& input(const std::string& arg_name, int32_t index) const {
    return conf_.input(arg_name, index);
  }

  int32_t output_size(const std::string& arg_name) const { return conf_.output_size(arg_name); }
  const std::string& output(const std::string& arg_name, int32_t index) const {
    return conf_.output(arg_name, index);
  }
  std::string output_grad(const std::string& output_arg_name, int32_t index) const;
  std::string GetGradTensorWithOpOutput(const std::string& output_arg_name, int32_t index) const;

  template<typename T>
  T attr(const std::string& attr_name) const {
    return conf_.attr<T>(attr_name);
  }

  const TensorDesc& arg_tensor_desc(const std::string& arg_name, int32_t index) const;
  const TensorDesc& TensorDesc4ArgNameAndIndex(const std::string& arg_name, int32_t index) const;

 public:
  void InputGradBind(const user_op::OpArg& input, const UserOpInputGradGetFn& grad_fn);
  void BindGradTensorWithOpInput(const std::string& logical_grad_blob_name,
                                 const std::string& input_arg_name, int32_t index) const;
  bool NeedGenGradTensor4OpInput(const std::string& input_arg_name, int32_t index) const;
  bool HasGradTensor4OpOutput(const std::string& output_arg_name, int32_t index) const;

 private:
  UserOpConfWrapper conf_;
  std::function<LogicalBlobId*(const std::string&)> diff_fn_;
  HashMap<std::string, TensorDesc> bn2tensor_desc_;
};

class UserOpConfWrapperBuilder final {
 public:
  UserOpConfWrapperBuilder(const std::string& op_name) : op_name_(op_name) {}

  UserOpConfWrapperBuilder& OpTypeName(const std::string& op_type_name) {
    op_type_name_ = op_type_name;
    return *this;
  }
  UserOpConfWrapperBuilder& Op(const std::string& op_type_name) { return OpTypeName(op_type_name); }

  UserOpConfWrapperBuilder& InputBind(const std::string& arg_name,
                                      const std::string& logical_blob_name);
  UserOpConfWrapperBuilder& Input(const std::string& arg_name,
                                  const std::string& logical_blob_name);

  UserOpConfWrapperBuilder& Output(const std::string& arg_name, int32_t num);
  UserOpConfWrapperBuilder& Output(const std::string& arg_name);

  template<typename T>
  UserOpConfWrapperBuilder& Attr(const std::string& attr_name, const T& val);

  UserOpConfWrapperBuilder& ScopeSymbolId(int64_t scope_symbol_id);

  UserOpConfWrapper Build();

 private:
  UserOpConfWrapper wrapper_;
  std::string op_name_;
  std::string op_type_name_;
  HashMap<std::string, std::vector<std::string>> input_;
  HashMap<std::string, std::vector<std::string>> output_;
  HashMap<std::string, AttrValue> attr_;
  OptInt64 scope_symbol_id_;
};

using BackwardOpBuilder = UserOpConfWrapperBuilder;
using BackwardOpBuilderFn = std::function<UserOpConfWrapper(BackwardOpBuilder&)>;
class BackwardOpConfContext final {
 public:
  BackwardOpConfContext(const UserOpWrapper& fw_op_wp, std::vector<OperatorConf>* bw_op_confs)
      : fw_op_wp_(fw_op_wp), bw_op_confs_(bw_op_confs) {}

 public:
  UserOpWrapper& FwOp() { return fw_op_wp_; }
  void DefineOp(const std::string& op_name, const BackwardOpBuilderFn& fn);
  UserOpConfWrapper& GetOp(const std::string& op_name);

 private:
  UserOpWrapper fw_op_wp_;
  HashMap<std::string, BackwardOpBuilderFn> op_builder_fns_;
  HashMap<std::string, UserOpConfWrapper> op_builder_results_;
  std::vector<OperatorConf>* bw_op_confs_;
};

}  // namespace user_op

Maybe<long long> GetAttrTypeImpl(const std::string& op_type_name, const std::string& attr_name);
Maybe<OperatorConf> CheckAndCompleteUserOpConfImpl(const OperatorConf& op_conf);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_USER_OP_CONF_H_
