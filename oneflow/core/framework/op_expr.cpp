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
#include "oneflow/user/kernels/stateful_local_opkernel.h"

namespace oneflow {
namespace one {

BuiltinOpExpr::BuiltinOpExpr(const std::string& op_name,
                             const std::vector<std::string>& indexed_ibns,
                             const std::vector<std::string>& indexed_obns)
    : op_name_(op_name),
      input_arg_tuple_(new ArgTuple(indexed_ibns)),
      output_arg_tuple_(new ArgTuple(indexed_obns)) {}

#define DEFINE_OPEXPR_TYPE_NAME(_T, _type_name)                \
  template<>                                                   \
  const std::string BuiltinOpExprImpl<_T>::type_name() const { \
    return _type_name;                                         \
  }

DEFINE_OPEXPR_TYPE_NAME(UserOpConf, "user");
DEFINE_OPEXPR_TYPE_NAME(VariableOpConf, "variable");
DEFINE_OPEXPR_TYPE_NAME(CastToMirroredOpConf, "cast_to_mirrored");
DEFINE_OPEXPR_TYPE_NAME(CastFromMirroredOpConf, "cast_from_mirrored");
DEFINE_OPEXPR_TYPE_NAME(DistributeSplitOpConf, "distribute_split");
DEFINE_OPEXPR_TYPE_NAME(DistributeCloneOpConf, "distribute_clone");
DEFINE_OPEXPR_TYPE_NAME(DistributeConcatOpConf, "distribute_concat");
DEFINE_OPEXPR_TYPE_NAME(DistributeAddOpConf, "distribute_add");

#undef DEFINE_OPEXPR_TYPE_NAME

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
      JUST(StatefulLocalOpKernel::New(op_conf, base_attrs(), device.mem_case(), parallel_desc,
                                      input_arg_tuple(), output_arg_tuple()));
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

class UserOpExprDeviceInferContext final : public user_op::DeviceInferContext {
 public:
  UserOpExprDeviceInferContext(const UserOpExpr* user_op_expr, const AttrMap& attrs,
                               const TensorTuple& input_tensors,
                               std::vector<std::shared_ptr<const Device>>* output_devices)
      : user_op_expr_(user_op_expr),
        composed_attrs_(attrs, user_op_expr->base_attrs()),
        input_tensors_(&input_tensors),
        output_devices_(output_devices) {}

  const std::vector<std::pair<std::string, int32_t>>& inputs() const override {
    return user_op_expr_->indexed_input_pairs();
  }

  const std::vector<std::pair<std::string, int32_t>>& outputs() const override {
    return user_op_expr_->indexed_output_pairs();
  }

  std::shared_ptr<const Device>* OutputTensorDevice4ArgNameAndIndex(const std::string& name,
                                                                    int64_t index) override {
    const auto& arg_tuple = *user_op_expr_->output_arg_tuple();
    std::size_t tuple_index = arg_tuple.TensorTupleIndex4ArgNameAndIndex(name, index);
    CHECK_GE(tuple_index, 0);
    return &output_devices_->at(tuple_index);
  }

  const std::shared_ptr<const Device>& InputTensorDevice4ArgNameAndIndex(
      const std::string& name, int64_t index) const override {
    const auto& arg_tuple = *user_op_expr_->input_arg_tuple();
    std::size_t tuple_index = arg_tuple.TensorTupleIndex4ArgNameAndIndex(name, index);
    CHECK_GE(tuple_index, 0);
    return input_tensors_->at(tuple_index)->device();
  }

 private:
  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override {
    return composed_attrs_.Attr4Name(attr_name);
  }
  const UserOpExpr* user_op_expr_;
  const ComposedAttrMap composed_attrs_;
  const TensorTuple* input_tensors_;
  std::vector<std::shared_ptr<const Device>>* output_devices_;
};

}  // namespace

UserOpExpr::UserOpExpr(const std::string& op_name, UserOpConf&& proto, const AttrMap& base_attrs,
                       const std::vector<std::string>& indexed_ibns,
                       const std::vector<std::string>& indexed_obns)
    : BuiltinOpExprImpl<UserOpConf>(op_name, std::move(proto), indexed_ibns, indexed_obns),
      base_attrs_(base_attrs) {
  const auto* registry =
      user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(op_proto_.op_type_name());
  if (registry && registry->device_infer_fn) { device_infer_fn_ = registry->device_infer_fn; }
}

/* static */ Maybe<UserOpExpr> UserOpExpr::New(const std::string& op_name, UserOpConf&& op_proto,
                                               const std::vector<std::string>& indexed_ibns,
                                               const std::vector<std::string>& indexed_obns) {
  AttrMap base_attrs = MakeAttrMapFromUserOpConf(op_proto);
  return std::shared_ptr<UserOpExpr>(
      new UserOpExpr(op_name, std::move(op_proto), base_attrs, indexed_ibns, indexed_obns));
}

Maybe<const Device> UserOpExpr::InferDevices(
    const AttrMap& attrs, const TensorTuple& input_tensors,
    std::vector<std::shared_ptr<const Device>>* output_devices) const {
  CHECK_OR_RETURN(static_cast<bool>(device_infer_fn_));
  UserOpExprDeviceInferContext device_infer_ctx(this, attrs, input_tensors, output_devices);
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
Maybe<bool> BuiltinOpExprImpl<VariableOpConf>::IsGradDisabled() const {
  return true;
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
