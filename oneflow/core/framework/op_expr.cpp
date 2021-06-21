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
#include "oneflow/core/framework/op_expr.h"

#include "oneflow/core/common/auto_registration_factory.h"
#include "oneflow/core/framework/attr_value_accessor.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/user/kernels/stateful_local_opkernel.h"

namespace oneflow {
namespace one {

BuiltinOpExpr::BuiltinOpExpr(const std::string& op_name,
                             const std::vector<std::string>& indexed_ibns,
                             const std::vector<std::string>& indexed_obns)
    : op_name_(op_name),
      input_arg_tuple_(new ArgTuple(indexed_ibns)),
      output_arg_tuple_(new ArgTuple(indexed_obns)) {}

#define DEFINE_OPEXPR_OP_TYPE_NAME(_T, _op_type_name)              \
  template<>                                                       \
  const std::string& BuiltinOpExprImpl<_T>::op_type_name() const { \
    static const std::string& name(_op_type_name);                 \
    return name;                                                   \
  }

DEFINE_OPEXPR_OP_TYPE_NAME(VariableOpConf, "variable");
DEFINE_OPEXPR_OP_TYPE_NAME(CastToMirroredOpConf, "cast_to_mirrored");
DEFINE_OPEXPR_OP_TYPE_NAME(CastFromMirroredOpConf, "cast_from_mirrored");
DEFINE_OPEXPR_OP_TYPE_NAME(DistributeSplitOpConf, "distribute_split");
DEFINE_OPEXPR_OP_TYPE_NAME(DistributeCloneOpConf, "distribute_clone");
DEFINE_OPEXPR_OP_TYPE_NAME(DistributeConcatOpConf, "distribute_concat");
DEFINE_OPEXPR_OP_TYPE_NAME(DistributeAddOpConf, "distribute_add");

#undef DEFINE_OPEXPR_OP_TYPE_NAME

template<>
const std::string& BuiltinOpExprImpl<UserOpConf>::op_type_name() const {
  return op_proto_.op_type_name();
}

#define DEFINE_OPEXPR_IS_GRAD_DISABLED_DEFAULT_VALUE(_T, _bool) \
  template<>                                                    \
  Maybe<bool> BuiltinOpExprImpl<_T>::IsGradDisabled() const {   \
    return _bool;                                               \
  }

DEFINE_OPEXPR_IS_GRAD_DISABLED_DEFAULT_VALUE(VariableOpConf, true);
DEFINE_OPEXPR_IS_GRAD_DISABLED_DEFAULT_VALUE(CastToMirroredOpConf, false);
DEFINE_OPEXPR_IS_GRAD_DISABLED_DEFAULT_VALUE(CastFromMirroredOpConf, false);
DEFINE_OPEXPR_IS_GRAD_DISABLED_DEFAULT_VALUE(DistributeSplitOpConf, false);
DEFINE_OPEXPR_IS_GRAD_DISABLED_DEFAULT_VALUE(DistributeCloneOpConf, false);
DEFINE_OPEXPR_IS_GRAD_DISABLED_DEFAULT_VALUE(DistributeConcatOpConf, false);
DEFINE_OPEXPR_IS_GRAD_DISABLED_DEFAULT_VALUE(DistributeAddOpConf, false);

#undef DEFINE_OPEXPR_IS_GRAD_DISABLED_DEFAULT_VALUE

template<>
Maybe<void> BuiltinOpExprImpl<UserOpConf>::BuildOpConf(OperatorConf* op_conf,
                                                       const AttrMap& attrs) const {
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_user_conf()) = op_proto_;
  auto* user_op_conf = op_conf->mutable_user_conf();
  for (const auto& it : attrs) {
    AttrValue attr_val;
    user_op::AttrValueUtil::ToProtoAttrValue(*it.second, &attr_val);
    (*(user_op_conf->mutable_attr()))[it.first] = attr_val;
  }
  return Maybe<void>::Ok();
}

Maybe<StatefulLocalOpKernel> UserOpExpr::MutKernel4Device(const Device& device) const {
  const auto& it = device2kernel_.find(device);
  if (it != device2kernel_.end()) { return it->second; }

  std::shared_ptr<OperatorConf> op_conf = std::make_shared<OperatorConf>();
  BuildOpConf(op_conf.get(), {});
  op_conf->set_device_tag(JUST(device.of_type()));
  std::shared_ptr<const ParallelDesc> parallel_desc = device.parallel_desc_ptr();
  const auto& opkernel =
      JUST(StatefulLocalOpKernel::New(op_conf, device.shared_from_this(), base_attrs(),
                                      parallel_desc, input_arg_tuple(), output_arg_tuple()));
  device2kernel_.emplace(device, opkernel);
  return opkernel;
}

template<>
Maybe<bool> BuiltinOpExprImpl<UserOpConf>::IsGradDisabled() const {
  const std::string& op_type_name = op_proto_.op_type_name();
  const user_op::OpGradRegistryResult* val =
      user_op::UserOpRegistryMgr::Get().GetOpGradRegistryResult(op_type_name);
  if (val) { return false; }
  return !IsClassRegistered<std::string, OpExprGradFunctionIf>(op_type_name);
}

template<>
Maybe<OpExprGradClosure> BuiltinOpExprImpl<UserOpConf>::GetOrCreateOpGradClosure() const {
  if (!op_grad_func_.get()) {
    if (IsClassRegistered<std::string, OpExprGradFunctionIf>(proto().op_type_name())) {
      op_grad_func_.reset(NewObj<std::string, OpExprGradFunctionIf>(proto().op_type_name()));
    } else {
      op_grad_func_.reset(NewObj<std::string, OpExprGradFunctionIf>("default"));
    }
    CHECK_NOTNULL_OR_RETURN(op_grad_func_.get());
    JUST(op_grad_func_->Init(*this));
  }
  return std::make_shared<OpExprGradClosure>(op_grad_func_);
}

namespace {

class UserOpExprInferContext : public user_op::InferContext {
 public:
  UserOpExprInferContext(const UserOpExpr* user_op_expr, const AttrMap& attrs,
                         const std::string& device_tag,
                         const std::function<const TensorMeta*(int32_t)>& TensorMeta4InputIndex,
                         const std::function<TensorMeta*(int32_t)>& TensorMeta4OutputIndex)
      : user_op_expr_(user_op_expr),
        composed_attrs_(attrs, user_op_expr->base_attrs()),
        device_tag_(device_tag),
        tensor_meta4input_index_(TensorMeta4InputIndex),
        tensor_meta4output_index_(TensorMeta4OutputIndex) {}
  ~UserOpExprInferContext() = default;

  const std::vector<std::pair<std::string, int32_t>>& inputs() const override {
    return user_op_expr_->indexed_input_pairs();
  }

  const std::vector<std::pair<std::string, int32_t>>& outputs() const override {
    return user_op_expr_->indexed_output_pairs();
  }

  user_op::TensorDesc* OutputTensorDesc(const std::string& name, int32_t index) override {
    return TensorDesc4ArgNameAndIndex(name, index);
  }

  user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& name, int32_t index) override {
    {
      const auto& arg_tuple = *user_op_expr_->output_arg_tuple();
      int32_t tuple_index = arg_tuple.TensorTupleIndex4ArgNameAndIndex(name, index);
      if (tuple_index >= 0) { return tensor_meta4output_index_(tuple_index); }
    }
    {
      const auto& arg_tuple = *user_op_expr_->input_arg_tuple();
      int32_t tuple_index = arg_tuple.TensorTupleIndex4ArgNameAndIndex(name, index);
      if (tuple_index >= 0) {
        return const_cast<TensorMeta*>(tensor_meta4input_index_(tuple_index));
      }
    }
    return nullptr;
  }

  const Shape& InputShape(const std::string& name, int32_t index) const override {
    const auto& arg_tuple = *user_op_expr_->input_arg_tuple();
    int32_t tuple_index = arg_tuple.TensorTupleIndex4ArgNameAndIndex(name, index);
    CHECK_GE(tuple_index, 0);
    return tensor_meta4input_index_(tuple_index)->shape();
  }

  Shape* OutputShape(const std::string& name, int32_t index) override {
    const auto& arg_tuple = *user_op_expr_->output_arg_tuple();
    int32_t tuple_index = arg_tuple.TensorTupleIndex4ArgNameAndIndex(name, index);
    CHECK_GE(tuple_index, 0);
    return tensor_meta4output_index_(tuple_index)->mut_shape();
  }

  Shape* Shape4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return TensorDesc4ArgNameAndIndex(arg_name, index)->mut_shape();
  }
  const DataType& InputDType(const std::string& arg_name, int32_t index) const override {
    return *const_cast<UserOpExprInferContext*>(this)->Dtype4ArgNameAndIndex(arg_name, index);
  }
  DataType* OutputDType(const std::string& arg_name, int32_t index) override {
    return const_cast<UserOpExprInferContext*>(this)->Dtype4ArgNameAndIndex(arg_name, index);
  }
  DataType* Dtype4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return TensorDesc4ArgNameAndIndex(arg_name, index)->mut_data_type();
  }
  bool InputIsDynamic4ArgNameAndIndex(const std::string& arg_name, int32_t index) const override {
    return *const_cast<UserOpExprInferContext*>(this)->IsDynamic4ArgNameAndIndex(arg_name, index);
  }
  bool* OutputIsDynamic4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return IsDynamic4ArgNameAndIndex(arg_name, index);
  }
  bool* IsDynamic4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return TensorDesc4ArgNameAndIndex(arg_name, index)->mut_is_dynamic();
  }
  const std::string& input(const std::string& arg_name, int32_t index) const override {
    const auto& arg_tuple = *user_op_expr_->input_arg_tuple();
    int32_t tuple_index = arg_tuple.TensorTupleIndex4ArgNameAndIndex(arg_name, index);
    CHECK_GE(tuple_index, 0);
    return arg_tuple.indexed_bns().at(tuple_index);
  }
  const std::string& output(const std::string& arg_name, int32_t index) const override {
    const auto& arg_tuple = *user_op_expr_->output_arg_tuple();
    int32_t tuple_index = arg_tuple.TensorTupleIndex4ArgNameAndIndex(arg_name, index);
    CHECK_GE(tuple_index, 0);
    return arg_tuple.indexed_bns().at(tuple_index);
  }
  bool has_input(const std::string& arg_name, int32_t index) const override {
    const auto& arg_tuple = *user_op_expr_->input_arg_tuple();
    int32_t tuple_index = arg_tuple.TensorTupleIndex4ArgNameAndIndex(arg_name, index);
    return tuple_index >= 0;
  }
  bool has_output(const std::string& arg_name, int32_t index) const override {
    const auto& arg_tuple = *user_op_expr_->output_arg_tuple();
    int32_t tuple_index = arg_tuple.TensorTupleIndex4ArgNameAndIndex(arg_name, index);
    return tuple_index >= 0;
  }
  int32_t input_size(const std::string& arg_name) const override {
    const auto& arg_tuple = *user_op_expr_->input_arg_tuple();
    return arg_tuple.arg_name2bn_index2tensor_tuple_index().at(arg_name).size();
  }
  int32_t output_size(const std::string& arg_name) const override {
    const auto& arg_tuple = *user_op_expr_->output_arg_tuple();
    return arg_tuple.arg_name2bn_index2tensor_tuple_index().at(arg_name).size();
  }
  const std::string& op_name() const override { return user_op_expr_->op_name(); }
  const std::string& op_type_name() const override { return user_op_expr_->op_type_name(); }
  const std::string& device_tag() const override { return device_tag_; }

  const user_op::TensorDesc* LogicalTensorDesc4ArgNameAndIndex(const std::string& name,
                                                               int32_t index) const override {
    UNIMPLEMENTED();
    return nullptr;
  }

  const ParallelContext& parallel_ctx() const override {
    UNIMPLEMENTED();
    return *(const ParallelContext*)nullptr;
  }
  const ParallelDesc& parallel_desc() const override {
    UNIMPLEMENTED();
    return *(const ParallelDesc*)nullptr;
  }
  const cfg::SbpParallel& SbpParallel4ArgNameAndIndex(const std::string&, int32_t) const override {
    UNIMPLEMENTED();
    return *(const cfg::SbpParallel*)nullptr;
  }
  const cfg::ParallelDistribution& ParallelDistribution4ArgNameAndIndex(const std::string&,
                                                                        int32_t) const override {
    UNIMPLEMENTED();
    return *(const cfg::ParallelDistribution*)nullptr;
  }
  int64_t parallel_num() const override {
    UNIMPLEMENTED();
    return 1;
  }

 private:
  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override {
    return composed_attrs_.Attr4Name(attr_name);
  }
  const UserOpExpr* user_op_expr_;
  const ComposedAttrMap composed_attrs_;
  const std::string& device_tag_;
  const std::function<const TensorMeta*(int32_t)>& tensor_meta4input_index_;
  const std::function<TensorMeta*(int32_t)>& tensor_meta4output_index_;
};

class UserOpExprDeviceInferContext final : public user_op::DeviceInferContext {
 public:
  UserOpExprDeviceInferContext(const UserOpExpr* user_op_expr, const AttrMap& attrs,
                               const TensorTuple& input_tensors, TensorTuple* output_tensors)
      : user_op_expr_(user_op_expr),
        composed_attrs_(attrs, user_op_expr->base_attrs()),
        input_tensors_(&input_tensors),
        output_tensors_(output_tensors) {}

  const std::vector<std::pair<std::string, int32_t>>& inputs() const override {
    return user_op_expr_->indexed_input_pairs();
  }

  const std::vector<std::pair<std::string, int32_t>>& outputs() const override {
    return user_op_expr_->indexed_output_pairs();
  }

  std::shared_ptr<const Device>* OutputTensorDevice4ArgNameAndIndex(const std::string& name,
                                                                    int64_t index) override {
    const auto& arg_tuple = *user_op_expr_->output_arg_tuple();
    int32_t tuple_index = arg_tuple.TensorTupleIndex4ArgNameAndIndex(name, index);
    CHECK_GE(tuple_index, 0);
    return CHECK_JUST(output_tensors_->at(tuple_index)->mut_device());
  }

  std::shared_ptr<const Device> InputTensorDevice4ArgNameAndIndex(const std::string& name,
                                                                  int64_t index) const override {
    const auto& arg_tuple = *user_op_expr_->input_arg_tuple();
    int32_t tuple_index = arg_tuple.TensorTupleIndex4ArgNameAndIndex(name, index);
    CHECK_GE(tuple_index, 0);
    return CHECK_JUST(input_tensors_->at(tuple_index)->device());
  }

 private:
  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override {
    return composed_attrs_.Attr4Name(attr_name);
  }
  const UserOpExpr* user_op_expr_;
  const ComposedAttrMap composed_attrs_;
  const TensorTuple* input_tensors_;
  TensorTuple* output_tensors_;
};

}  // namespace

UserOpExpr::UserOpExpr(const std::string& op_name, UserOpConf&& proto, const AttrMap& base_attrs,
                       const std::vector<std::string>& indexed_ibns,
                       const std::vector<std::string>& indexed_obns)
    : BuiltinOpExprImpl<UserOpConf>(op_name, std::move(proto), indexed_ibns, indexed_obns),
      base_attrs_(base_attrs) {}

Maybe<void> UserOpExpr::Init() {
  const auto* registry =
      user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(op_proto_.op_type_name());
  CHECK_NOTNULL_OR_RETURN(registry);
  shape_infer_fn_ = registry->logical_tensor_desc_infer_fn;
  CHECK_OR_RETURN(static_cast<bool>(shape_infer_fn_));
  dtype_infer_fn_ = registry->data_type_infer_fn;
  CHECK_OR_RETURN(static_cast<bool>(dtype_infer_fn_));
  if (registry->device_infer_fn) { device_infer_fn_ = registry->device_infer_fn; }
  return Maybe<void>::Ok();
}

/* static */ Maybe<UserOpExpr> UserOpExpr::New(const std::string& op_name, UserOpConf&& op_proto,
                                               const std::vector<std::string>& indexed_ibns,
                                               const std::vector<std::string>& indexed_obns) {
  AttrMap base_attrs = MakeAttrMapFromUserOpConf(op_proto);
  std::shared_ptr<UserOpExpr> op_expr(
      new UserOpExpr(op_name, std::move(op_proto), base_attrs, indexed_ibns, indexed_obns));
  JUST(op_expr->Init());
  return op_expr;
}

Maybe<void> UserOpExpr::InferLogicalShapeAndDType(
    const AttrMap& attrs, const std::string& device_tag,
    const std::function<const TensorMeta*(int32_t)>& TensorMeta4InputIndex,
    const std::function<TensorMeta*(int32_t)>& TensorMeta4OutputIndex) const {
  UserOpExprInferContext infer_ctx(this, attrs, device_tag, TensorMeta4InputIndex,
                                   TensorMeta4OutputIndex);
  JUST(shape_infer_fn_(&infer_ctx));
  JUST(dtype_infer_fn_(&infer_ctx));
  return Maybe<void>::Ok();
}

Maybe<const Device> UserOpExpr::InferDevices(const AttrMap& attrs, const TensorTuple& input_tensors,
                                             TensorTuple* output_tensors) const {
  CHECK_OR_RETURN(static_cast<bool>(device_infer_fn_));
  UserOpExprDeviceInferContext device_infer_ctx(this, attrs, input_tensors, output_tensors);
  return TRY(device_infer_fn_(&device_infer_ctx));
}

template<>
Maybe<void> BuiltinOpExprImpl<VariableOpConf>::BuildOpConf(OperatorConf* op_conf,
                                                           const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(attrs.size(), 0);
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_variable_conf()) = op_proto_;
  return Maybe<void>::Ok();
}

template<>
Maybe<OpExprGradClosure> BuiltinOpExprImpl<VariableOpConf>::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED_THEN_RETURN();
}

template<>
Maybe<void> BuiltinOpExprImpl<CastToMirroredOpConf>::BuildOpConf(OperatorConf* op_conf,
                                                                 const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(attrs.size(), 0);
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_cast_to_mirrored_conf()) = op_proto_;
  return Maybe<void>::Ok();
}

template<>
Maybe<OpExprGradClosure> BuiltinOpExprImpl<CastToMirroredOpConf>::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED_THEN_RETURN();
}

template<>
Maybe<void> BuiltinOpExprImpl<CastFromMirroredOpConf>::BuildOpConf(OperatorConf* op_conf,
                                                                   const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(attrs.size(), 0);
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_cast_from_mirrored_conf()) = op_proto_;
  return Maybe<void>::Ok();
}

template<>
Maybe<OpExprGradClosure> BuiltinOpExprImpl<CastFromMirroredOpConf>::GetOrCreateOpGradClosure()
    const {
  UNIMPLEMENTED_THEN_RETURN();
}

template<>
Maybe<void> BuiltinOpExprImpl<DistributeSplitOpConf>::BuildOpConf(OperatorConf* op_conf,
                                                                  const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(attrs.size(), 0);
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_distribute_split_conf()) = op_proto_;
  return Maybe<void>::Ok();
}

template<>
Maybe<OpExprGradClosure> BuiltinOpExprImpl<DistributeSplitOpConf>::GetOrCreateOpGradClosure()
    const {
  UNIMPLEMENTED_THEN_RETURN();
}

template<>
Maybe<void> BuiltinOpExprImpl<DistributeCloneOpConf>::BuildOpConf(OperatorConf* op_conf,
                                                                  const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(attrs.size(), 0);
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_distribute_clone_conf()) = op_proto_;
  return Maybe<void>::Ok();
}

template<>
Maybe<OpExprGradClosure> BuiltinOpExprImpl<DistributeCloneOpConf>::GetOrCreateOpGradClosure()
    const {
  UNIMPLEMENTED_THEN_RETURN();
}

template<>
Maybe<void> BuiltinOpExprImpl<DistributeConcatOpConf>::BuildOpConf(OperatorConf* op_conf,
                                                                   const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(attrs.size(), 0);
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_distribute_concat_conf()) = op_proto_;
  return Maybe<void>::Ok();
}

template<>
Maybe<OpExprGradClosure> BuiltinOpExprImpl<DistributeConcatOpConf>::GetOrCreateOpGradClosure()
    const {
  UNIMPLEMENTED_THEN_RETURN();
}

template<>
Maybe<void> BuiltinOpExprImpl<DistributeAddOpConf>::BuildOpConf(OperatorConf* op_conf,
                                                                const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(attrs.size(), 0);
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_distribute_add_conf()) = op_proto_;
  return Maybe<void>::Ok();
}

template<>
Maybe<OpExprGradClosure> BuiltinOpExprImpl<DistributeAddOpConf>::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED_THEN_RETURN();
}

}  // namespace one
}  // namespace oneflow
