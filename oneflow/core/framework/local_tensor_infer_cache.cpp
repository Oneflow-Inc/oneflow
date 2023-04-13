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
#include "oneflow/core/framework/local_tensor_infer_cache.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/env_var/eager.h"
#include "oneflow/core/framework/infer_util.h"

namespace oneflow {
namespace one {

namespace {

Maybe<void> CheckIsDeviceSupportedByOp(const Device& device, const std::string& op_type_name) {
  if (IsCpuOnly(op_type_name)) { CHECK_EQ_OR_RETURN(device.type(), "cpu"); }  // NOLINT
  return Maybe<void>::Ok();
}

Maybe<void> CheckInputDeviceIdentical(const LocalTensorMetaInferArgs& infer_args,
                                      Symbol<Device> default_device,
                                      const UserOpExpr& user_op_expr) {
  for (int i = 0; i < infer_args.input_local_tensor_metas().size(); ++i) {
    if (user_op_expr.IsHostMemoryInput(i)) { continue; }
    CHECK_OR_RETURN(default_device
                    == JUST(VectorAt(infer_args.input_local_tensor_metas(), i))->device())
        << Error::RuntimeError()
        << "Expected all tensors to be on the same device, but found "
           "at least two devices, "
        << default_device->ToString() << " (positional 0) and "
        << JUST(VectorAt(infer_args.input_local_tensor_metas(), i))->device()->ToString()
        << " (positional " << i << ")!";
  }
  return Maybe<void>::Ok();
}

class UserOpExprDeviceAndStreamInferContext final : public user_op::DeviceAndStreamInferContext {
 public:
  UserOpExprDeviceAndStreamInferContext(const UserOpExpr* user_op_expr,
                                        const LocalTensorMetaInferArgs& infer_args,
                                        OpArgsVector<MutLocalTensorMeta>* output_tensor_metas)
      : user_op_expr_(user_op_expr),
        composed_attrs_(infer_args.attrs(), user_op_expr->base_attrs()),
        infer_args_(infer_args),
        output_tensor_metas_(output_tensor_metas) {}

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
    CHECK_GE(tuple_index, 0) << "tuple index should be non-negative, but got " << tuple_index;
    CHECK_LT(tuple_index, user_op_expr_->output_size())
        << "tuple index " << tuple_index << " should be less than output size "
        << user_op_expr_->output_size();
    return output_tensor_metas_->at(tuple_index).mut_device();
  }

  Symbol<Device> InputTensorDevice4ArgNameAndIndex(const std::string& name,
                                                   int64_t index) const override {
    const auto& arg_tuple = *user_op_expr_->input_arg_tuple();
    int32_t tuple_index = arg_tuple.TensorTupleIndex4ArgNameAndIndex(name, index);
    CHECK_GE(tuple_index, 0) << "tuple index should be non-negative, but got " << tuple_index;
    CHECK_LT(tuple_index, user_op_expr_->input_size())
        << "tuple index " << tuple_index << " should be less than input size "
        << user_op_expr_->input_size();
    return infer_args_.input_local_tensor_metas().at(tuple_index)->device();
  }

 private:
  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override {
    return composed_attrs_.Attr4Name(attr_name);
  }
  const UserOpExpr* user_op_expr_;
  const ComposedAttrMap composed_attrs_;
  const LocalTensorMetaInferArgs& infer_args_;
  OpArgsVector<MutLocalTensorMeta>* output_tensor_metas_;
};

Maybe<Symbol<Stream>> InferDeviceAndStream(const UserOpExpr& user_op_expr,
                                           const Symbol<Device>& default_device,
                                           const LocalTensorMetaInferArgs& infer_args,
                                           OpArgsVector<MutLocalTensorMeta>* output_tensor_metas) {
  Symbol<Stream> stream;
  if (!user_op_expr.has_device_and_stream_infer_fn()) {
    stream = JUST(GetDefaultStreamByDevice(default_device));
    for (int i = 0; i < user_op_expr.output_size(); i++) {
      auto& tensor_meta = output_tensor_metas->at(i);
      *tensor_meta.mut_device() = default_device;
    }
  } else {
    if (!user_op_expr.device_and_stream_infer_fn()) {
      Symbol<Device> device = infer_args.input_local_tensor_metas().at(0)->device();
      stream = JUST(GetDefaultStreamByDevice(device));
    } else {
      UserOpExprDeviceAndStreamInferContext device_and_stream_ctx(&user_op_expr, infer_args,
                                                                  output_tensor_metas);
      stream = JUST(user_op_expr.device_and_stream_infer_fn()(&device_and_stream_ctx));
    }
  }
  return stream;
}

}  // namespace

size_t LocalTensorMetaInferArgs::hash_value() const {
  size_t hash_value = std::hash<AttrMap>()(attrs_);
  HashCombine(&hash_value, std::hash<Symbol<Device>>()(default_device_));
  const auto& tensor_meta_hash_functor = std::hash<Symbol<LocalTensorMeta>>();
  for (const auto& tensor_meta : input_local_tensor_metas_) {
    HashCombine(&hash_value, tensor_meta_hash_functor(tensor_meta));
  }
  return hash_value;
}

bool LocalTensorMetaInferArgs::operator==(const LocalTensorMetaInferArgs& other) const {
  return this->attrs_ == other.attrs_ && this->default_device_ == other.default_device_
         && this->input_local_tensor_metas_ == other.input_local_tensor_metas_;
}

Maybe<void> LocalTensorMetaInferArgs::Init(const AttrMap& attrs, Symbol<Device> default_device,
                                           const TensorTuple& input_tensors) {
  this->attrs_ = attrs;
  this->default_device_ = default_device;
  this->input_local_tensor_metas_.resize(input_tensors.size());
  JUST(this->InitInputLocalTensorMetas(input_tensors));
  return Maybe<void>::Ok();
}

Maybe<void> LocalTensorMetaInferArgs::InitInputLocalTensorMetas(const TensorTuple& input_tensors) {
  for (int i = 0; i < input_tensors.size(); ++i) {
    input_local_tensor_metas_.at(i) = JUST(input_tensors.at(i)->local_tensor_meta());
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<const LocalTensorInferResult> LocalTensorInferCache::Infer(
    const UserOpExpr& user_op_expr, const LocalTensorMetaInferArgs& infer_args) {
  const auto& default_device = infer_args.default_device();
  JUST(CheckInputDeviceIdentical(infer_args, default_device, user_op_expr));
  JUST(CheckIsDeviceSupportedByOp(*default_device, user_op_expr.op_type_name()));

  auto result = std::make_unique<LocalTensorInferResult>(user_op_expr.output_size());

  OpArgsVector<MutLocalTensorMeta> output_mut_metas(user_op_expr.output_size());
  // Infer devices
  Symbol<Stream> stream =
      JUST(InferDeviceAndStream(user_op_expr, default_device, infer_args, &output_mut_metas));
  result->set_stream(stream);

  {
    const auto& GetInputTensorMeta = [&](int32_t i) -> const TensorMeta* {
      return infer_args.input_local_tensor_metas().at(i).shared_from_symbol().get();
    };
    JUST(user_op_expr.InferPhysicalTensorDesc(
        infer_args.attrs(), stream->device()->type(), GetInputTensorMeta,
        [&](int32_t i) -> TensorMeta* { return &output_mut_metas.at(i); }));
  }

  auto* mut_output_tensor_metas = result->mut_output_tensor_metas();
  for (int32_t i = 0; i < user_op_expr.output_size(); ++i) {
    if (!JUST(user_op_expr.SupportNonContiguous())) {
      Stride stride(output_mut_metas.at(i).shape());
      output_mut_metas.at(i).set_stride(stride);
    }
    CHECK_OR_RETURN(static_cast<bool>(output_mut_metas.at(i).device()))
        << Error::RuntimeError() << "device not infered";
    mut_output_tensor_metas->at(i) = SymbolOf(
        LocalTensorMeta(output_mut_metas.at(i).shape(), output_mut_metas.at(i).stride(),
                        output_mut_metas.at(i).data_type(), output_mut_metas.at(i).device()));
  }
  return std::shared_ptr<const LocalTensorInferResult>(std::move(result));
}

Maybe<const LocalTensorInferResult> LocalTensorInferCache::GetOrInfer(
    const LocalTensorMetaInferArgs& infer_args) {
  if (ThreadLocalEnvBool<ONEFLOW_EAGER_ENABLE_LOCAL_INFER_CACHE>()) {
    auto iter = cache_.find(infer_args);
    if (iter == cache_.end()) {
      if (unlikely(cache_.size()
                   >= ThreadLocalEnvInteger<ONEFLOW_EAGER_TENSOR_INFER_CACHE_SIZE>())) {
        cache_.clear();
      }
      const auto& user_op_expr = user_op_expr_.lock();
      CHECK_OR_RETURN(static_cast<bool>(user_op_expr));  // NOLINT
      const auto& output_tensor_metas = JUST(Infer(*user_op_expr, infer_args));
      iter = cache_.emplace(infer_args, output_tensor_metas).first;
    }
    return iter->second;
  } else {
    const auto& user_op_expr = user_op_expr_.lock();
    return JUST(Infer(*user_op_expr, infer_args));
  }
}

}  // namespace one
}  // namespace oneflow
