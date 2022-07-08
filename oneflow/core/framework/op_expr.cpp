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
#include <memory>
#include "oneflow/core/common/error.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/common/auto_registration_factory.h"
#include "oneflow/core/framework/attr_value_accessor.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/op_interpreter/dispatch_frame.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/core/framework/consistent_tensor_infer_cache.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/user/kernels/stateful_opkernel.h"

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

DEFINE_OPEXPR_OP_TYPE_NAME(FeedInputOpConf, "feed_input");
DEFINE_OPEXPR_OP_TYPE_NAME(FeedVariableOpConf, "feed_variable");
DEFINE_OPEXPR_OP_TYPE_NAME(FetchOutputOpConf, "fetch_output");
DEFINE_OPEXPR_OP_TYPE_NAME(ImageDecoderRandomCropResizeOpConf, "image_gpu_decode");
DEFINE_OPEXPR_OP_TYPE_NAME(VariableOpConf, "variable");
DEFINE_OPEXPR_OP_TYPE_NAME(CastToLocalOpConf, "cast_to_local");
DEFINE_OPEXPR_OP_TYPE_NAME(CastFromLocalOpConf, "cast_from_local");
DEFINE_OPEXPR_OP_TYPE_NAME(DistributeSplitOpConf, "distribute_split");
DEFINE_OPEXPR_OP_TYPE_NAME(DistributeCloneOpConf, "distribute_clone");
DEFINE_OPEXPR_OP_TYPE_NAME(DistributeConcatOpConf, "distribute_concat");
DEFINE_OPEXPR_OP_TYPE_NAME(DistributeAddOpConf, "distribute_add");

#undef DEFINE_OPEXPR_OP_TYPE_NAME

template<>
const std::string& BuiltinOpExprImpl<UserOpConf>::op_type_name() const {
  return op_proto_.op_type_name();
}

const std::string& ConsistentToConsistentOpExpr::op_type_name() const {
  static const std::string kOpTypeName = "consistent_to_consistent";
  return kOpTypeName;
}

const std::string& CastToConsistentOpExpr::op_type_name() const {
  static const std::string kOpTypeName = "cast_to_consistent";
  return kOpTypeName;
}

const std::string& CastFromConsistentOpExpr::op_type_name() const {
  static const std::string kOpTypeName = "cast_from_consistent";
  return kOpTypeName;
}

#define DEFINE_OPEXPR_IS_GRAD_DISABLED_AND_SUPPORT_NON_CONTIGUOUS_DEFAULT_VALUE(_T, flag) \
  template<>                                                                              \
  Maybe<bool> BuiltinOpExprImpl<_T>::IsGradDisabled() const {                             \
    return flag;                                                                          \
  }                                                                                       \
  template<>                                                                              \
  Maybe<bool> BuiltinOpExprImpl<_T>::SupportNonContiguous() const {                       \
    return false;                                                                         \
  }

DEFINE_OPEXPR_IS_GRAD_DISABLED_AND_SUPPORT_NON_CONTIGUOUS_DEFAULT_VALUE(FeedInputOpConf, true);
DEFINE_OPEXPR_IS_GRAD_DISABLED_AND_SUPPORT_NON_CONTIGUOUS_DEFAULT_VALUE(FeedVariableOpConf, true);
DEFINE_OPEXPR_IS_GRAD_DISABLED_AND_SUPPORT_NON_CONTIGUOUS_DEFAULT_VALUE(FetchOutputOpConf, true);
DEFINE_OPEXPR_IS_GRAD_DISABLED_AND_SUPPORT_NON_CONTIGUOUS_DEFAULT_VALUE(VariableOpConf, true);
DEFINE_OPEXPR_IS_GRAD_DISABLED_AND_SUPPORT_NON_CONTIGUOUS_DEFAULT_VALUE(
    ImageDecoderRandomCropResizeOpConf, true);
DEFINE_OPEXPR_IS_GRAD_DISABLED_AND_SUPPORT_NON_CONTIGUOUS_DEFAULT_VALUE(CastToLocalOpConf, false);
DEFINE_OPEXPR_IS_GRAD_DISABLED_AND_SUPPORT_NON_CONTIGUOUS_DEFAULT_VALUE(CastFromLocalOpConf, false);
DEFINE_OPEXPR_IS_GRAD_DISABLED_AND_SUPPORT_NON_CONTIGUOUS_DEFAULT_VALUE(DistributeSplitOpConf,
                                                                        false);
DEFINE_OPEXPR_IS_GRAD_DISABLED_AND_SUPPORT_NON_CONTIGUOUS_DEFAULT_VALUE(DistributeCloneOpConf,
                                                                        false);
DEFINE_OPEXPR_IS_GRAD_DISABLED_AND_SUPPORT_NON_CONTIGUOUS_DEFAULT_VALUE(DistributeConcatOpConf,
                                                                        false);
DEFINE_OPEXPR_IS_GRAD_DISABLED_AND_SUPPORT_NON_CONTIGUOUS_DEFAULT_VALUE(DistributeAddOpConf, false);

#undef DEFINE_OPEXPR_IS_GRAD_DISABLED_AND_SUPPORT_NON_CONTIGUOUS_DEFAULT_VALUE

template<>
Maybe<void> BuiltinOpExprImpl<UserOpConf>::BuildOpConf(OperatorConf* op_conf,
                                                       const AttrMap& attrs) const {
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_user_conf()) = op_proto_;
  *(op_conf->mutable_loc()) = DispatchFrame::get_str();
  auto* user_op_conf = op_conf->mutable_user_conf();
  for (const auto& it : attrs) {
    AttrValue attr_val;
    JUST(user_op::AttrValueUtil::ToProtoAttrValue(*it.second, &attr_val));
    (*(user_op_conf->mutable_attr()))[it.first] = attr_val;
  }
  return Maybe<void>::Ok();
}

Maybe<StatefulOpKernel> UserOpExpr::MutKernel4Stream(Symbol<Stream> stream) const {
  const auto& it = stream2kernel_.find(stream);
  if (it != stream2kernel_.end()) { return it->second; }

  std::shared_ptr<OperatorConf> op_conf = std::make_shared<OperatorConf>();
  JUST(BuildOpConf(op_conf.get(), {}));
  op_conf->set_device_tag(stream->device()->type());
  auto parallel_desc = JUST(Placement4Device(stream->device())).shared_from_symbol();
  const auto& opkernel = JUST(StatefulOpKernel::New(op_conf, stream, base_attrs(), parallel_desc,
                                                    input_arg_tuple(), output_arg_tuple()));
  stream2kernel_.emplace(stream, opkernel);
  return opkernel;
}

template<>
Maybe<bool> BuiltinOpExprImpl<UserOpConf>::IsGradDisabled() const {
  const auto* registry =
      user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(proto().op_type_name());
  CHECK_NOTNULL_OR_RETURN(registry);
  return registry->no_grad;
}

template<>
Maybe<bool> BuiltinOpExprImpl<UserOpConf>::SupportNonContiguous() const {
  const auto* registry =
      user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(proto().op_type_name());
  CHECK_NOTNULL_OR_RETURN(registry)
      << "The op(operation) " << proto().op_type_name()
      << " is not found. Please check whether it has been registered correctly.";
  return registry->non_contiguous_supported;
}

template<>
Maybe<OpExprGradClosure> BuiltinOpExprImpl<UserOpConf>::GetOrCreateOpGradClosure() const {
  if (!op_grad_func_.get()) {
    CHECK_OR_RETURN((IsClassRegistered<std::string, OpExprGradFunctionIf>(proto().op_type_name())))
        << "The gradient function for op " << proto().op_type_name()
        << " is not found. Please check whether it has been implemented and registered correctly.";
    op_grad_func_.reset(NewObj<std::string, OpExprGradFunctionIf>(proto().op_type_name()));
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
        tensor_meta4input_index_(TensorMeta4InputIndex),
        tensor_meta4output_index_(TensorMeta4OutputIndex) {
    loc_ = DispatchFrame::get_str();
  }
  virtual ~UserOpExprInferContext() override = default;

  const std::vector<std::pair<std::string, int32_t>>& inputs() const override {
    return user_op_expr_->indexed_input_pairs();
  }

  const std::vector<std::pair<std::string, int32_t>>& outputs() const override {
    return user_op_expr_->indexed_output_pairs();
  }

  const user_op::TensorDesc& InputTensorDesc(const std::string& arg_name,
                                             int32_t index) const override {
    return *const_cast<UserOpExprInferContext*>(this)->TensorDesc4ArgNameAndIndex(arg_name, index);
  }

  user_op::TensorDesc* OutputTensorDesc(const std::string& name, int32_t index) override {
    return TensorDesc4ArgNameAndIndex(name, index);
  }

  user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& name, int32_t index) {
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

  const Stride& InputStride(const std::string& name, int32_t index) const override {
    const auto& arg_tuple = *user_op_expr_->input_arg_tuple();
    int32_t tuple_index = arg_tuple.TensorTupleIndex4ArgNameAndIndex(name, index);
    CHECK_GE(tuple_index, 0);
    return tensor_meta4input_index_(tuple_index)->stride();
  }

  Stride* OutputStride(const std::string& name, int32_t index) override {
    const auto& arg_tuple = *user_op_expr_->output_arg_tuple();
    int32_t tuple_index = arg_tuple.TensorTupleIndex4ArgNameAndIndex(name, index);
    CHECK_GE(tuple_index, 0);
    return tensor_meta4output_index_(tuple_index)->mut_stride();
  }

  Stride* Stride4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return TensorDesc4ArgNameAndIndex(arg_name, index)->mut_stride();
  }

  const DataType& InputDType(const std::string& arg_name, int32_t index) const override {
    return *const_cast<UserOpExprInferContext*>(this)->Dtype4ArgNameAndIndex(arg_name, index);
  }
  DataType* OutputDType(const std::string& arg_name, int32_t index) override {
    return Dtype4ArgNameAndIndex(arg_name, index);
  }
  DataType* Dtype4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return TensorDesc4ArgNameAndIndex(arg_name, index)->mut_data_type();
  }
  bool InputIsDynamic(const std::string& arg_name, int32_t index) const override {
    return *const_cast<UserOpExprInferContext*>(this)->IsDynamic4ArgNameAndIndex(arg_name, index);
  }
  bool* OutputIsDynamic(const std::string& arg_name, int32_t index) override {
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
  const std::string& op_loc() const override { return loc_; }

 private:
  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override {
    return composed_attrs_.Attr4Name(attr_name);
  }
  const UserOpExpr* user_op_expr_;
  const ComposedAttrMap composed_attrs_;
  const std::function<const TensorMeta*(int32_t)>& tensor_meta4input_index_;
  const std::function<TensorMeta*(int32_t)>& tensor_meta4output_index_;
  std::string loc_;
};

class UserOpExprPhysicalInferContext final : public UserOpExprInferContext {
 public:
  using UserOpExprInferContext::UserOpExprInferContext;
  ~UserOpExprPhysicalInferContext() override = default;

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
  const SbpParallel& SbpParallel4ArgNameAndIndex(const std::string&, int32_t) const override {
    UNIMPLEMENTED();
    return *(const SbpParallel*)nullptr;
  }
  const NdSbp& NdSbp4ArgNameAndIndex(const std::string&, int32_t) const override {
    UNIMPLEMENTED();
    return *(const NdSbp*)nullptr;
  }
  int64_t parallel_num() const override { return 1; }
};

class UserOpExprLogicalInferContext final : public UserOpExprInferContext {
 public:
  UserOpExprLogicalInferContext(
      const UserOpExpr* user_op_expr, const AttrMap& attrs, Symbol<ParallelDesc> parallel_desc,
      const std::function<const TensorMeta*(int32_t)>& TensorMeta4InputIndex,
      const std::function<TensorMeta*(int32_t)>& TensorMeta4OutputIndex)
      : UserOpExprInferContext(user_op_expr, attrs, parallel_desc->device_tag(),
                               TensorMeta4InputIndex, TensorMeta4OutputIndex),
        parallel_desc_(parallel_desc) {
    const auto& opt_parallel_id = CHECK_JUST(GetParallelId4CurrentProcessCtx(parallel_desc_));
    // Default parallel_id = -1, which will not cause bad effects becauce it will never be used in
    // LogicalTensorDescInfer.
    int64_t parallel_id = -1;
    if (opt_parallel_id->has_value()) { parallel_id = CHECK_JUST(*opt_parallel_id); }
    parallel_ctx_.set_parallel_id(parallel_id);
    parallel_ctx_.set_parallel_num(parallel_desc_->parallel_num());
  }
  ~UserOpExprLogicalInferContext() override = default;

  const user_op::TensorDesc* LogicalTensorDesc4ArgNameAndIndex(const std::string& name,
                                                               int32_t index) const override {
    UNIMPLEMENTED();
  }

  const ParallelContext& parallel_ctx() const override { return parallel_ctx_; }
  const ParallelDesc& parallel_desc() const override { return *parallel_desc_; }
  const SbpParallel& SbpParallel4ArgNameAndIndex(const std::string& name,
                                                 int32_t index) const override {
    auto* tensor_meta = dynamic_cast<ConsistentTensorMeta*>(
        const_cast<UserOpExprLogicalInferContext*>(this)->TensorDesc4ArgNameAndIndex(name, index));
    CHECK_NOTNULL(tensor_meta);
    Symbol<NdSbp> nd_sbp = tensor_meta->nd_sbp();
    CHECK_EQ(nd_sbp->sbp_parallel_size(), 1);
    return nd_sbp->sbp_parallel(0);
  }
  const NdSbp& NdSbp4ArgNameAndIndex(const std::string& name, int32_t index) const override {
    auto* tensor_meta = dynamic_cast<ConsistentTensorMeta*>(
        const_cast<UserOpExprLogicalInferContext*>(this)->TensorDesc4ArgNameAndIndex(name, index));
    CHECK_NOTNULL(tensor_meta);
    return *tensor_meta->nd_sbp();
  }
  int64_t parallel_num() const override { return parallel_desc_->parallel_num(); }

 private:
  Symbol<ParallelDesc> parallel_desc_;
  ParallelContext parallel_ctx_;
};

class UserOpExprDeviceAndStreamInferContext final : public user_op::DeviceAndStreamInferContext {
 public:
  UserOpExprDeviceAndStreamInferContext(const UserOpExpr* user_op_expr, const AttrMap& attrs,
                                        const TensorTuple& input_tensors,
                                        TensorTuple* output_tensors)
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

  Symbol<Device>* OutputTensorDevice4ArgNameAndIndex(const std::string& name,
                                                     int64_t index) override {
    const auto& arg_tuple = *user_op_expr_->output_arg_tuple();
    int32_t tuple_index = arg_tuple.TensorTupleIndex4ArgNameAndIndex(name, index);
    CHECK_GE(tuple_index, 0);
    return CHECK_JUST(output_tensors_->at(tuple_index)->mut_device());
  }

  Symbol<Device> InputTensorDevice4ArgNameAndIndex(const std::string& name,
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

Maybe<void> UserOpExpr::Init(const std::shared_ptr<const UserOpExpr>& self) {
  const auto* registry =
      user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(op_proto_.op_type_name());
  CHECK_NOTNULL_OR_RETURN(registry);
  tensor_desc_infer_fn_ = registry->logical_tensor_desc_infer_fn;
  CHECK_OR_RETURN(static_cast<bool>(tensor_desc_infer_fn_))
      << Error::RuntimeError() << "registry->logical_tensor_desc_infer_fn failed.";
  dtype_infer_fn_ = registry->data_type_infer_fn;
  CHECK_OR_RETURN(static_cast<bool>(dtype_infer_fn_))
      << Error::RuntimeError() << "registry->data_type_infer_fn failed.";
  if (registry->device_and_stream_infer_fn) {
    device_and_stream_infer_fn_ = registry->device_and_stream_infer_fn;
  }
  consistent_tensor_infer_cache_.reset(new ConsistentTensorInferCache(self));
  return Maybe<void>::Ok();
}

/* static */ Maybe<UserOpExpr> UserOpExpr::New(const std::string& op_name, UserOpConf&& op_proto,
                                               const std::vector<std::string>& indexed_ibns,
                                               const std::vector<std::string>& indexed_obns) {
  JUST(AddAttrDefaultValueAndCheckValid(&op_proto));
  AttrMap base_attrs = MakeAttrMapFromUserOpConf(op_proto);
  std::shared_ptr<UserOpExpr> op_expr(
      new UserOpExpr(op_name, std::move(op_proto), base_attrs, indexed_ibns, indexed_obns));
  JUST(op_expr->Init(op_expr));
  return op_expr;
}

Maybe<void> UserOpExpr::InferPhysicalTensorDesc(
    const AttrMap& attrs, const std::string& device_tag,
    const std::function<const TensorMeta*(int32_t)>& TensorMeta4InputIndex,
    const std::function<TensorMeta*(int32_t)>& TensorMeta4OutputIndex) const {
  UserOpExprPhysicalInferContext infer_ctx(this, attrs, device_tag, TensorMeta4InputIndex,
                                           TensorMeta4OutputIndex);
  JUST(tensor_desc_infer_fn_(&infer_ctx));
  JUST(dtype_infer_fn_(&infer_ctx));
  return Maybe<void>::Ok();
}

Maybe<void> UserOpExpr::InferLogicalTensorDesc(
    const AttrMap& attrs, Symbol<ParallelDesc> parallel_desc,
    const std::function<const TensorMeta*(int32_t)>& TensorMeta4InputIndex,
    const std::function<TensorMeta*(int32_t)>& TensorMeta4OutputIndex) const {
  UserOpExprLogicalInferContext infer_ctx(this, attrs, parallel_desc, TensorMeta4InputIndex,
                                          TensorMeta4OutputIndex);
  JUST(tensor_desc_infer_fn_(&infer_ctx));
  JUST(dtype_infer_fn_(&infer_ctx));
  return Maybe<void>::Ok();
}

Maybe<Symbol<Stream>> UserOpExpr::InferDeviceAndStream(const AttrMap& attrs,
                                                       const TensorTuple& input_tensors,
                                                       TensorTuple* output_tensors) const {
  CHECK_OR_RETURN(static_cast<bool>(device_and_stream_infer_fn_));
  UserOpExprDeviceAndStreamInferContext device_infer_ctx(this, attrs, input_tensors,
                                                         output_tensors);
  return TRY(device_and_stream_infer_fn_(&device_infer_ctx));
}

ConsistentToConsistentOpExpr::ConsistentToConsistentOpExpr(
    const Optional<Symbol<NdSbp>>& grad_nd_sbp)
    : grad_nd_sbp_(grad_nd_sbp) {}

/* static */ Maybe<ConsistentToConsistentOpExpr> ConsistentToConsistentOpExpr::New(
    const Optional<Symbol<NdSbp>>& grad_nd_sbp) {
  auto* ptr = new ConsistentToConsistentOpExpr(grad_nd_sbp);
  return std::shared_ptr<ConsistentToConsistentOpExpr>(ptr);
}

CastConsistentOpExpr::CastConsistentOpExpr(const std::string& op_name) : op_name_(op_name) {}

CastToConsistentOpExpr::CastToConsistentOpExpr(const std::string& op_name)
    : CastConsistentOpExpr(op_name) {}

/* static */ Maybe<CastToConsistentOpExpr> CastToConsistentOpExpr::New(const std::string& op_name) {
  return std::shared_ptr<CastToConsistentOpExpr>(new CastToConsistentOpExpr(op_name));
}

CastFromConsistentOpExpr::CastFromConsistentOpExpr(const std::string& op_name)
    : CastConsistentOpExpr(op_name) {}

/* static */ Maybe<CastFromConsistentOpExpr> CastFromConsistentOpExpr::New(
    const std::string& op_name) {
  return std::shared_ptr<CastFromConsistentOpExpr>(new CastFromConsistentOpExpr(op_name));
}

template<>
Maybe<void> BuiltinOpExprImpl<FeedInputOpConf>::BuildOpConf(OperatorConf* op_conf,
                                                            const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(attrs.size(), 0);
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_feed_input_conf()) = op_proto_;
  *(op_conf->mutable_loc()) = DispatchFrame::get_str();
  return Maybe<void>::Ok();
}

template<>
Maybe<OpExprGradClosure> BuiltinOpExprImpl<FeedInputOpConf>::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED_THEN_RETURN();
}

template<>
Maybe<void> BuiltinOpExprImpl<FeedVariableOpConf>::BuildOpConf(OperatorConf* op_conf,
                                                               const AttrMap& attrs) const {
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_feed_variable_conf()) = op_proto_;
  *(op_conf->mutable_loc()) = DispatchFrame::get_str();
  return Maybe<void>::Ok();
}

template<>
Maybe<OpExprGradClosure> BuiltinOpExprImpl<FeedVariableOpConf>::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED_THEN_RETURN();
}

template<>
Maybe<void> BuiltinOpExprImpl<FetchOutputOpConf>::BuildOpConf(OperatorConf* op_conf,
                                                              const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(attrs.size(), 0);
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_fetch_output_conf()) = op_proto_;
  *(op_conf->mutable_loc()) = DispatchFrame::get_str();
  return Maybe<void>::Ok();
}

template<>
Maybe<OpExprGradClosure> BuiltinOpExprImpl<FetchOutputOpConf>::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED_THEN_RETURN();
}

template<>
Maybe<void> BuiltinOpExprImpl<ImageDecoderRandomCropResizeOpConf>::BuildOpConf(
    OperatorConf* op_conf, const AttrMap& attrs) const {
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_image_decoder_random_crop_resize_conf()) = op_proto_;
  *(op_conf->mutable_loc()) = DispatchFrame::get_str();
  auto* proto = op_conf->mutable_image_decoder_random_crop_resize_conf();
  proto->set_target_width(JUST(attrs.GetAttr<int64_t>("target_width")));
  proto->set_target_height(JUST(attrs.GetAttr<int64_t>("target_height")));
  proto->set_num_workers(JUST(attrs.GetAttr<int64_t>("num_workers")));
  proto->set_max_num_pixels(JUST(attrs.GetAttr<int64_t>("max_num_pixels")));
  proto->set_warmup_size(JUST(attrs.GetAttr<int64_t>("warmup_size")));
  proto->set_seed(JUST(attrs.GetAttr<int64_t>("seed")));
  proto->set_num_attempts(JUST(attrs.GetAttr<int64_t>("num_attempts")));
  proto->set_random_area_min(JUST(attrs.GetAttr<float>("random_area_min")));
  proto->set_random_area_max(JUST(attrs.GetAttr<float>("random_area_max")));
  proto->set_random_aspect_ratio_min(JUST(attrs.GetAttr<float>("random_aspect_ratio_min")));
  proto->set_random_aspect_ratio_max(JUST(attrs.GetAttr<float>("random_aspect_ratio_max")));
  return Maybe<void>::Ok();
}

template<>
Maybe<OpExprGradClosure>
BuiltinOpExprImpl<ImageDecoderRandomCropResizeOpConf>::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED_THEN_RETURN();
}

template<>
Maybe<void> BuiltinOpExprImpl<VariableOpConf>::BuildOpConf(OperatorConf* op_conf,
                                                           const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(attrs.size(), 0);
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_variable_conf()) = op_proto_;
  *(op_conf->mutable_loc()) = DispatchFrame::get_str();
  return Maybe<void>::Ok();
}

template<>
Maybe<OpExprGradClosure> BuiltinOpExprImpl<VariableOpConf>::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED_THEN_RETURN();
}

template<>
Maybe<void> BuiltinOpExprImpl<CastToLocalOpConf>::BuildOpConf(OperatorConf* op_conf,
                                                              const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(attrs.size(), 0);
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_cast_to_local_conf()) = op_proto_;
  *(op_conf->mutable_loc()) = DispatchFrame::get_str();
  return Maybe<void>::Ok();
}

template<>
Maybe<OpExprGradClosure> BuiltinOpExprImpl<CastToLocalOpConf>::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED_THEN_RETURN();
}

template<>
Maybe<void> BuiltinOpExprImpl<CastFromLocalOpConf>::BuildOpConf(OperatorConf* op_conf,
                                                                const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(attrs.size(), 0);
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_cast_from_local_conf()) = op_proto_;
  *(op_conf->mutable_loc()) = DispatchFrame::get_str();
  return Maybe<void>::Ok();
}

template<>
Maybe<OpExprGradClosure> BuiltinOpExprImpl<CastFromLocalOpConf>::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED_THEN_RETURN();
}

Maybe<OpExprGradClosure> ConsistentToConsistentOpExpr::GetOrCreateOpGradClosure() const {
  if (!op_grad_func_.get()) {
    op_grad_func_.reset(NewObj<std::string, OpExprGradFunctionIf>("consistent_to_consistent"));
    CHECK_NOTNULL_OR_RETURN(op_grad_func_.get());
    JUST(op_grad_func_->Init(*this));
  }
  return std::make_shared<OpExprGradClosure>(op_grad_func_);
}

Maybe<OpExprGradClosure> CastToConsistentOpExpr::GetOrCreateOpGradClosure() const {
  if (!op_grad_func_.get()) {
    op_grad_func_.reset(NewObj<std::string, OpExprGradFunctionIf>("cast_to_consistent"));
    CHECK_NOTNULL_OR_RETURN(op_grad_func_.get());
    JUST(op_grad_func_->Init(*this));
  }
  return std::make_shared<OpExprGradClosure>(op_grad_func_);
}

Maybe<OpExprGradClosure> CastFromConsistentOpExpr::GetOrCreateOpGradClosure() const {
  if (!op_grad_func_.get()) {
    op_grad_func_.reset(NewObj<std::string, OpExprGradFunctionIf>("cast_from_consistent"));
    CHECK_NOTNULL_OR_RETURN(op_grad_func_.get());
    JUST(op_grad_func_->Init(*this));
  }
  return std::make_shared<OpExprGradClosure>(op_grad_func_);
}

template<>
Maybe<void> BuiltinOpExprImpl<DistributeSplitOpConf>::BuildOpConf(OperatorConf* op_conf,
                                                                  const AttrMap& attrs) const {
  CHECK_EQ_OR_RETURN(attrs.size(), 0);
  *(op_conf->mutable_name()) = op_name_;
  *(op_conf->mutable_distribute_split_conf()) = op_proto_;
  *(op_conf->mutable_loc()) = DispatchFrame::get_str();
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
  *(op_conf->mutable_loc()) = DispatchFrame::get_str();
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
  *(op_conf->mutable_loc()) = DispatchFrame::get_str();
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
  *(op_conf->mutable_loc()) = DispatchFrame::get_str();
  return Maybe<void>::Ok();
}

template<>
Maybe<OpExprGradClosure> BuiltinOpExprImpl<DistributeAddOpConf>::GetOrCreateOpGradClosure() const {
  UNIMPLEMENTED_THEN_RETURN();
}

Maybe<OpExprGradClosure> SelectTopNOpExpr::GetOrCreateOpGradClosure() const {
  if (!op_grad_func_.get()) {
    op_grad_func_.reset(NewObj<std::string, OpExprGradFunctionIf>("select_top_n"));
    CHECK_NOTNULL_OR_RETURN(op_grad_func_.get());
    JUST(op_grad_func_->Init(*this));
  }
  return std::make_shared<OpExprGradClosure>(op_grad_func_);
}

void FunctionOpExpr::reset_state() const { state_.reset(new FunctionAutoGradCaptureState); }

Maybe<OpExprGradClosure> FunctionOpExpr::GetOrCreateOpGradClosure() const {
  if (!op_grad_func_) { op_grad_func_.reset(new FunctionOpExprGradFunction(backward_fn_)); }
  return std::make_shared<OpExprGradClosure>(op_grad_func_, state_);
}

}  // namespace one
}  // namespace oneflow
