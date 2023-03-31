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
#include "oneflow/core/autograd/autograd_mode.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/mutable_attr_map.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/placement_utils.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/functional/sequence_function.h"
#include "oneflow/core/functional/impl/unary_functor.h"
#include "oneflow/core/ep/include/device_manager_registry.h"
#include "oneflow/core/job/global_mode.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/framework/tensor_util.h"
#include "oneflow/core/job/nd_sbp_util.h"
#include <complex>

namespace oneflow {
namespace one {
namespace functional {
namespace impl {

class ArgMaxFunctor {
 public:
  ArgMaxFunctor() { op_ = CHECK_JUST(one::OpBuilder("argmax").Input("in").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const Optional<int32_t>& dim,
                           const Optional<bool>& keepdim,
                           const Optional<Symbol<DType>>& dtype) const {
    if (dim.has_value() == false) {
      return SequenceFunction<Maybe<Tensor>()>([&]() { return Flatten(input, 0, -1); })
          .then([&](const std::shared_ptr<one::Tensor>& x) {
            return OpInterpUtil::Dispatch<Tensor>(*op_, {x});
          })
          .call();
    }

    int new_dim = JUST(dim);
    const int32_t ndims = input->shape()->NumAxes();
    new_dim = JUST(maybe_wrap_dim(new_dim, ndims));
    if (new_dim < 0) { new_dim += ndims; }
    const auto do_cast = [&](const std::shared_ptr<one::Tensor>& x) -> Maybe<Tensor> {
      return Cast(x, JUST(dtype), /*pin_memory=*/false);
    };

    if (new_dim == ndims - 1) {
      return SequenceFunction<Maybe<Tensor>()>(
                 [&]() { return OpInterpUtil::Dispatch<Tensor>(*op_, {input}); })
          .then_if(keepdim.has_value() && JUST(keepdim) == true,
                   std::bind(ExpandDims, std::placeholders::_1, -1))
          .then_if(dtype.has_value(), do_cast)
          .call();
    }

    std::vector<int32_t> permute;
    permute.reserve(ndims);
    for (int32_t i = 0; i < ndims - 1; i++) { permute.emplace_back(i < new_dim ? i : i + 1); }
    permute.emplace_back(new_dim);

    std::vector<int32_t> permute_inv(ndims, 0);
    for (int32_t i = 0; i < ndims; i++) { permute_inv[i] = -1; }
    for (int32_t i = 0; i < ndims; i++) { permute_inv[permute[i]] = i; }

    std::vector<int32_t> squeeze_dim = {new_dim};

    return SequenceFunction<Maybe<Tensor>()>([&]() { return Transpose(input, permute); })
        .then([&](const std::shared_ptr<one::Tensor>& x) {
          return OpInterpUtil::Dispatch<Tensor>(*op_, {x});
        })
        .then(std::bind(ExpandDims, std::placeholders::_1, -1))
        .then(std::bind(Transpose, std::placeholders::_1, permute_inv))
        .then_if((!keepdim.has_value()) || (keepdim.has_value() && JUST(keepdim) == false),
                 std::bind(Squeeze, std::placeholders::_1, squeeze_dim))
        .then_if(dtype.has_value(), do_cast)
        .call();
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ArgMinFunctor {
 public:
  ArgMinFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const Optional<int32_t>& dim,
                           const Optional<bool>& keepdim,
                           const Optional<Symbol<DType>>& dtype) const {
    TensorProcessor tensor_processor;
    JUST(tensor_processor.AddInputs({input}, DType::Float()).Apply());
    const auto x = JUST(tensor_processor.GetInputs()).at(0);
    return sequence_function(Negative)
        .then(std::bind(ArgMax, std::placeholders::_1, dim, keepdim, dtype))
        .call(x);
  }
};

class GlobalTensorConstantFunctor {
 public:
  GlobalTensorConstantFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("tensor_constant").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const Shape& shape, const std::shared_ptr<one::Tensor>& value,
                           const Symbol<DType>& dtype, const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp_tuple) const {
    CHECK_OR_RETURN(value->ndim() <= 1 && value->nelement() == 1)
        << "Only tensor with single element or scalar tensor are supported as value!";
    CHECK_OR_RETURN(value->is_global()) << "The value tensor should be global tensor";
    // NOTE: this op is an source op, so the value(scalar tensor) should not have autograd status.
    autograd::AutoGradMode mode(false);
    JUST(CheckDeviceIdsIsValid(placement));
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("shape", "dtype", "nd_sbp");
    attrs.SetAllAttrs(shape, dtype->data_type(), NullOpt);

    auto dispatch_constant =
        [&](const std::vector<Symbol<SbpParallel>>& sbp_tuple) -> Maybe<Tensor> {
      std::vector<std::string> nd_sbp(sbp_tuple.size());
      {
        for (int i = 0; i < sbp_tuple.size(); ++i) {
          nd_sbp[i] = SbpParallelToString(*sbp_tuple[i]);
        }
      }
      attrs.SetAttr<2>(nd_sbp);
      return OpInterpUtil::Dispatch<Tensor>(*op_, {value}, attrs);
    };
    bool has_partial_parallel =
        std::any_of(sbp_tuple.begin(), sbp_tuple.end(),
                    [](const Symbol<SbpParallel>& sbp) { return sbp->has_partial_sum_parallel(); });
    // The source op does not support Partial
    if (has_partial_parallel) {
      const auto& fixed_sbp_tuple = JUST(NdSbpReplacePartialByBroadcast(sbp_tuple));
      const auto& tensor = JUST(dispatch_constant(*fixed_sbp_tuple));
      return functional::ToGlobal(tensor, placement, sbp_tuple, {}, /* check_meta */ false,
                                  /*copy*/ false);
    } else {
      return dispatch_constant(sbp_tuple);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class TensorConstantFunctor {
 public:
  TensorConstantFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("tensor_constant").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const Shape& shape, const std::shared_ptr<one::Tensor>& value,
                           const Symbol<DType>& dtype,
                           const Optional<Symbol<Device>>& device) const {
    CHECK_OR_RETURN(value->ndim() <= 1 && value->nelement() == 1)
        << "Only tensor with single element or scalar tensor are supported as value!";
    // NOTE: this op is an source op, so the value(scalar tensor) should not have autograd status.
    autograd::AutoGradMode mode(false);
    if (GlobalMode::is_enabled()) {
      auto global_mode_gurad = GlobalMode::Guard(false);
      return JUST(functional::GlobalTensorConstant(shape, value, dtype,
                                                   GetGlobalParallelDescFromDevice(device),
                                                   *JUST(GetSbpList(GlobalMode::nd_sbp()))));
    }
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("shape", "dtype");
    attrs.SetAllAttrs(shape, dtype->data_type());
    if (device.has_value()) {
      Symbol<Device> device_symbol = JUST(device);
      return OpInterpUtil::Dispatch<Tensor>(*op_, {value},
                                            OpExprInterpContext(attrs, device_symbol));
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*op_, {value}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class GlobalConstantFunctor {
 public:
  GlobalConstantFunctor() { op_ = CHECK_JUST(one::OpBuilder("constant").Output("out").Build()); }
  Maybe<Tensor> operator()(const Shape& shape, const Scalar& value, const Symbol<DType>& dtype,
                           const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp_tuple) const {
    JUST(CheckDeviceIdsIsValid(placement));
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("shape", "dtype", "complex_value",
                                                 "is_complex_value", "floating_value",
                                                 "is_floating_value", "integer_value", "nd_sbp");
    if (IsComplexDataType(dtype->data_type())) {
      attrs.SetAllAttrs(shape, dtype->data_type(), value.Value<std::complex<double>>(), true,
                        NullOpt, false, NullOpt, NullOpt);
    } else if (IsIntegralDataType(dtype->data_type())) {
      attrs.SetAllAttrs(shape, dtype->data_type(), NullOpt, false, NullOpt, false,
                        value.As<int64_t>(), NullOpt);
    } else {
      attrs.SetAllAttrs(shape, dtype->data_type(), NullOpt, false, value.As<double>(), true,
                        NullOpt, NullOpt);
    }

    auto dispatch_constant =
        [&](const std::vector<Symbol<SbpParallel>>& sbp_tuple) -> Maybe<Tensor> {
      if (LazyMode::is_enabled()) {
        std::vector<std::string> nd_sbp(sbp_tuple.size());
        {
          for (int i = 0; i < sbp_tuple.size(); ++i) {
            nd_sbp[i] = SbpParallelToString(*sbp_tuple[i]);
          }
        }
        attrs.SetAttr<7>(nd_sbp);
      }
      const auto& nd_sbp = JUST(GetNdSbp(sbp_tuple));
      return OpInterpUtil::Dispatch<Tensor>(*op_, {},
                                            OpExprInterpContext(attrs, placement, nd_sbp));
    };
    bool has_partial_parallel = [&]() {
      for (const auto& sbp : sbp_tuple) {
        if (sbp->has_partial_sum_parallel()) { return true; }
      }
      return false;
    }();
    // Since the source op does not support Partial, it is necessary to replace Partial
    // with Broadcast, and then convert it to Partial
    if (has_partial_parallel) {
      const auto& fixed_sbp_tuple = JUST(NdSbpReplacePartialByBroadcast(sbp_tuple));
      const auto& tensor = JUST(dispatch_constant(*fixed_sbp_tuple));
      return functional::ToGlobal(tensor, placement, sbp_tuple, {}, /* check_meta */ false,
                                  /*copy*/ false);
    } else {
      return dispatch_constant(sbp_tuple);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ConstantFunctor {
 public:
  ConstantFunctor() { op_ = CHECK_JUST(one::OpBuilder("constant").Output("out").Build()); }
  Maybe<Tensor> operator()(const Shape& shape, const Scalar& value, const Symbol<DType>& dtype,
                           const Optional<Symbol<Device>>& device) const {
    if (GlobalMode::is_enabled()) {
      auto global_mode_gurad = GlobalMode::Guard(false);
      return JUST(functional::GlobalConstant(shape, value, dtype,
                                             GetGlobalParallelDescFromDevice(device),
                                             *JUST(GetSbpList(GlobalMode::nd_sbp()))));
    }
    auto& attrs =
        THREAD_CACHED_MUTABLE_ATTR_MAP("shape", "dtype", "complex_value", "is_complex_value",
                                       "floating_value", "is_floating_value", "integer_value");
    if (IsComplexDataType(dtype->data_type())) {
      attrs.SetAllAttrs(shape, dtype->data_type(), value.Value<std::complex<double>>(), true,
                        NullOpt, false, NullOpt);
    } else if (IsIntegralDataType(dtype->data_type())) {
      attrs.SetAllAttrs(shape, dtype->data_type(), NullOpt, false, NullOpt, false,
                        value.As<int64_t>());
    } else {
      attrs.SetAllAttrs(shape, dtype->data_type(), NullOpt, false, value.As<double>(), true,
                        NullOpt);
    }
    if (device.has_value()) {
      Symbol<Device> device_symbol = JUST(device);
      return OpInterpUtil::Dispatch<Tensor>(*op_, {}, OpExprInterpContext(attrs, device_symbol));
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*op_, {}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class EmptyFunctor {
 public:
  EmptyFunctor() { op_ = CHECK_JUST(one::OpBuilder("empty").Output("out").Build()); }
  Maybe<Tensor> operator()(const Shape& shape, const Symbol<DType>& dtype,
                           const Optional<Symbol<Device>>& device, const bool requires_grad,
                           const bool pin_memory) const {
    std::shared_ptr<Tensor> empty;
    if (GlobalMode::is_enabled()) {
      auto global_mode_gurad = GlobalMode::Guard(false);
      empty = JUST(functional::GlobalEmpty(shape, dtype, GetGlobalParallelDescFromDevice(device),
                                           *JUST(GetSbpList(GlobalMode::nd_sbp()))));
      if (dtype->is_floating_point()) { JUST(empty->set_requires_grad(requires_grad)); }
      return empty;
    }
    Symbol<Device> device_symbol = device.value_or(JUST(Device::New("cpu", 0)));
    auto& attrs =
        THREAD_CACHED_MUTABLE_ATTR_MAP("shape", "dtype", "pin_memory", "device_type", "device_id");
    attrs.SetAllAttrs(shape, dtype->data_type(), pin_memory, device_symbol->type(),
                      device_symbol->device_id());
    if (device.has_value()) {
      Symbol<Device> device_symbol = JUST(device);
      empty =
          JUST(OpInterpUtil::Dispatch<Tensor>(*op_, {}, OpExprInterpContext(attrs, device_symbol)));
    } else {
      empty = JUST(OpInterpUtil::Dispatch<Tensor>(*op_, {}, attrs));
    }

    if (dtype->is_floating_point()) { JUST(empty->set_requires_grad(requires_grad)); }
    return empty;
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class EmptyStridedFunctor {
 public:
  Maybe<Tensor> operator()(const std::vector<int64_t>& shape, const std::vector<int64_t>& stride,
                           const Optional<Symbol<DType>>& dtype,
                           const Optional<Symbol<Device>>& device, const bool requires_grad,
                           const bool pin_memory) const {
    Symbol<DType> data_type = GetDefaultDType();
    if (dtype.has_value()) { data_type = JUST(dtype); }
    auto empty = JUST(functional::Empty(Shape(shape), dtype.value_or(GetDefaultDType()), device,
                                        requires_grad, pin_memory));
    CHECK_OR_RETURN(view::IsViewApplicable(empty))
        << "oneflow.empty_strided() only support in eager local mode!";
    return view::AsStrided(empty, shape, stride, 1);
  }
};

class GlobalEmptyFunctor {
 public:
  GlobalEmptyFunctor() { op_ = CHECK_JUST(one::OpBuilder("empty").Output("out").Build()); }
  Maybe<Tensor> operator()(const Shape& shape, const Symbol<DType>& dtype,
                           const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp_tuple) const {
    JUST(CheckDeviceIdsIsValid(placement));
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("shape", "dtype", "nd_sbp");
    if (LazyMode::is_enabled()) {
      std::vector<std::string> nd_sbp(sbp_tuple.size());
      {
        for (int i = 0; i < sbp_tuple.size(); ++i) {
          nd_sbp.at(i) = SbpParallelToString(*sbp_tuple.at(i));
        }
      }
      attrs.SetAllAttrs(shape, dtype->data_type(), nd_sbp);
    } else {
      attrs.SetAllAttrs(shape, dtype->data_type(), NullOpt);
    }
    const auto& nd_sbp = JUST(GetNdSbp(sbp_tuple));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {}, OpExprInterpContext(attrs, placement, nd_sbp));
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ZerosLikeFunctor : public UnaryFunctor {
 public:
  ZerosLikeFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("zero_like").Input("like").Output("out").Build());
  }
};

class OnesLikeFunctor : public UnaryFunctor {
 public:
  OnesLikeFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("ones_like").Input("like").Output("out").Build());
  }
};

class FullLikeFunctor {
 public:
  FullLikeFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Scalar& fill_value) const {
    std::shared_ptr<Tensor> out;
    if (x->is_local()) {
      out = JUST(functional::Empty(*(x->shape()), x->dtype(), JUST(x->device()),
                                   /*requires_grad=*/false, /*pin_memory=*/false));
    } else {
      out = JUST(functional::GlobalEmpty(*(x->shape()), x->dtype(), JUST(x->parallel_desc()),
                                         *JUST(private_details::RawGetSbpList(JUST(x->nd_sbp())))));
    }
    out = JUST(functional::Fill(out, fill_value));
    return out;
  }
};

class FlattenFunctor {
 public:
  FlattenFunctor() = default;

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const int32_t& start_dim,
                           const int32_t& end_dim) const {
    const Shape& in_shape = *x->shape();
    int32_t ndim = in_shape.size();

    auto CheckAndWrapDim = [&](int32_t dim) -> Maybe<int32_t> {
      // handle scalar
      if (ndim == 0 && (dim == 0 || dim == -1)) { return 0; }
      if (dim < -ndim || dim >= ndim) {
        return Error::IndexError() << "Dimension out of range (expected to be in range of ["
                                   << -ndim << ", " << ndim - 1 << "], but got " << dim << ")";
      }
      return dim >= 0 ? dim : dim + ndim;
    };

    // -n dim (negative dim) indicate ndim-n
    // for example, when ndim == 3, (-3) == (0), (-2) == (1), (-1) == (2)
    int32_t true_start_dim = JUST(CheckAndWrapDim(start_dim));
    int32_t true_end_dim = JUST(CheckAndWrapDim(end_dim));

    if (true_start_dim > true_end_dim) {
      return Error::RuntimeError() << "flatten() has invalid args: start_dim (" << start_dim
                                   << ") cannot come after end_dim (" << end_dim << ")";
    }

    // identity when start_dim == end_dim
    if (true_start_dim == true_end_dim) { return x; }

    DimVector dim_vec{in_shape.begin(), in_shape.begin() + true_start_dim + 1};
    for (int i = true_start_dim + 1; i <= true_end_dim; ++i) { dim_vec.back() *= in_shape[i]; }
    dim_vec.insert(dim_vec.end(), in_shape.begin() + true_end_dim + 1, in_shape.end());
    Shape reshape_shape{dim_vec};
    CHECK_EQ_OR_RETURN(in_shape.elem_cnt(), reshape_shape.elem_cnt())
        << Error::RuntimeError() << "invalid reshape from " << in_shape.ToString() << " to "
        << reshape_shape.ToString();
    return JUST(Reshape(x, reshape_shape));
  }
};

class WhereFunctor {
 public:
  WhereFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("where").Input("condition").Input("x").Input("y").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& condition,
                           const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {condition, x, y});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class WhereScalarXFunctor {
 public:
  WhereScalarXFunctor() = default;

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& condition, const Scalar& scalar,
                           const std::shared_ptr<one::Tensor>& y) const {
    std::shared_ptr<one::Tensor> x;
    if (y->is_local()) {
      x = JUST(functional::Constant(Shape({}), scalar, y->dtype(), JUST(y->device())));
    } else {
      const size_t sbp_ndim = JUST(y->nd_sbp())->sbp_parallel_size();
      std::vector<Symbol<SbpParallel>> nd_sbp_vec;
      nd_sbp_vec.reserve(sbp_ndim);
      for (int i = 0; i < sbp_ndim; ++i) {
        SbpParallel sbp;
        sbp.mutable_broadcast_parallel();
        nd_sbp_vec.push_back(SymbolOf(sbp));
      }
      const auto& parallel_desc = JUST(y->parallel_desc());
      x = JUST(
          functional::GlobalConstant(Shape({}), scalar, y->dtype(), parallel_desc, nd_sbp_vec));
    }
    return functional::Where(condition, x, y);
  }
};

class WhereScalarYFunctor {
 public:
  WhereScalarYFunctor() = default;

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& condition,
                           const std::shared_ptr<one::Tensor>& x, const Scalar& scalar) const {
    std::shared_ptr<one::Tensor> y;
    if (x->is_local()) {
      y = JUST(functional::Constant(Shape({}), scalar, x->dtype(), JUST(x->device())));
    } else {
      const size_t sbp_ndim = JUST(x->nd_sbp())->sbp_parallel_size();
      std::vector<Symbol<SbpParallel>> nd_sbp_vec;
      nd_sbp_vec.reserve(sbp_ndim);
      for (int i = 0; i < sbp_ndim; ++i) {
        SbpParallel sbp;
        sbp.mutable_broadcast_parallel();
        nd_sbp_vec.push_back(SymbolOf(sbp));
      }
      const auto& parallel_desc = JUST(x->parallel_desc());
      y = JUST(
          functional::GlobalConstant(Shape({}), scalar, x->dtype(), parallel_desc, nd_sbp_vec));
    }
    return functional::Where(condition, x, y);
  }
};

class WhereScalarXYFunctor {
 public:
  WhereScalarXYFunctor() = default;

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& condition, const Scalar& x_scalar,
                           const Scalar& y_scalar) const {
    std::shared_ptr<one::Tensor> x;
    std::shared_ptr<one::Tensor> y;
    DataType dtype = DataType::kInvalidDataType;

    if (x_scalar.IsBool() && y_scalar.IsBool()) {
      dtype = DataType::kBool;
    } else if (x_scalar.IsFloatingPoint() && y_scalar.IsFloatingPoint()) {
      double x_val = x_scalar.As<double>();
      double y_val = y_scalar.As<double>();
      if (x_val >= GetMinVal<DataTypeToType<DataType::kFloat>>()
          && x_val <= GetMaxVal<DataTypeToType<DataType::kFloat>>()
          && y_val >= GetMinVal<DataTypeToType<DataType::kFloat>>()
          && y_val <= GetMaxVal<DataTypeToType<DataType::kFloat>>()) {
        dtype = DataType::kFloat;
      } else {
        dtype = DataType::kDouble;
      }
    } else if (x_scalar.IsIntegral() && y_scalar.IsIntegral()) {
      if (x_scalar.IsUnsigned() && y_scalar.IsUnsigned()) {
        uint64_t x_val = x_scalar.As<uint64_t>();
        uint64_t y_val = y_scalar.As<uint64_t>();
        if (x_val <= GetMaxVal<DataTypeToType<DataType::kUInt32>>()
            && y_val <= GetMaxVal<DataTypeToType<DataType::kUInt32>>()) {
          dtype = DataType::kUInt32;
        } else {
          dtype = DataType::kUInt64;
        }
      } else if (x_scalar.IsSigned() && y_scalar.IsSigned()) {
        int64_t x_val = x_scalar.As<int64_t>();
        int64_t y_val = y_scalar.As<int64_t>();
        if (x_val >= GetMinVal<DataTypeToType<DataType::kInt32>>()
            && x_val <= GetMaxVal<DataTypeToType<DataType::kInt32>>()
            && y_val >= GetMinVal<DataTypeToType<DataType::kInt32>>()
            && y_val <= GetMaxVal<DataTypeToType<DataType::kInt32>>()) {
          dtype = DataType::kInt32;
        } else {
          dtype = DataType::kInt64;
        }
      } else {
        UNIMPLEMENTED_THEN_RETURN()
            << "The x scalar and y scalar in Where shoule be signed or unsigned at the same time.";
      }
    } else {
      UNIMPLEMENTED_THEN_RETURN()
          << "The x scalar and y in Where shoule be bool, float or int at the same time.";
    }

    if (condition->is_local()) {
      x = JUST(functional::Constant(Shape({}), x_scalar, DType(dtype), JUST(condition->device())));
      y = JUST(functional::Constant(Shape({}), y_scalar, DType(dtype), JUST(condition->device())));
    } else {
      const size_t sbp_ndim = JUST(condition->nd_sbp())->sbp_parallel_size();
      std::vector<Symbol<SbpParallel>> nd_sbp_vec;
      nd_sbp_vec.reserve(sbp_ndim);
      for (int i = 0; i < sbp_ndim; ++i) {
        SbpParallel sbp;
        sbp.mutable_broadcast_parallel();
        nd_sbp_vec.push_back(SymbolOf(sbp));
      }
      const auto& parallel_desc = JUST(condition->parallel_desc());
      x = JUST(
          functional::GlobalConstant(Shape({}), x_scalar, DType(dtype), parallel_desc, nd_sbp_vec));
      y = JUST(
          functional::GlobalConstant(Shape({}), y_scalar, DType(dtype), parallel_desc, nd_sbp_vec));
    }
    return functional::Where(condition, x, y);
  }
};

class ArgWhereFunctor {
 public:
  ArgWhereFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("argwhere").Input("input").Output("output").Output("output_size").Build());
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& x,
                                const Symbol<DType>& dtype) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("dtype");
    attrs.SetAllAttrs(dtype->data_type());
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class NonZeroFunctor {
 public:
  NonZeroFunctor() {}
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& x, bool as_tuple) const {
    std::shared_ptr<one::Tensor> input = x;
    if (as_tuple && input->ndim() == 0) { input = JUST(functional::Unsqueeze(input, 0)); }
    int64_t ndim = input->ndim();
    const auto& output_tuple =
        JUST(functional::ArgWhere(input, JUST(DType::Get(DataType::kInt64))));
    const std::shared_ptr<one::Tensor>& size = JUST(VectorAt(*output_tuple, 1));
    CHECK_EQ_OR_RETURN(size->shape()->elem_cnt(), 1)
        << Error::RuntimeError() << kOfBugIssueUploadPrompt;
    CHECK_OR_RETURN(size->dtype() == JUST(DType::Get(DataType::kInt64)))
        << Error::RuntimeError() << kOfBugIssueUploadPrompt;
    int64_t size_val = -1;
    {
      if (size->is_global()) {
        CHECK_OR_RETURN(JUST(size->parallel_desc())->parallel_num() == 1  // NOLINT
                        || NdSbpIsAllBroadcast(*JUST(size->nd_sbp())));   // NOLINT
      }
      JUST(GetItemInScalarTensor(size->is_local() ? size : JUST(size->cur_rank_phy_tensor()),
                                 &size_val, sizeof(size_val)));
    }
    std::vector<int64_t> start{0, 0};
    std::vector<int64_t> stop{size_val, ndim};
    std::vector<int64_t> step{1, 1};
    const auto& output = JUST(
        functional::Slice(output_tuple->at(0), start, stop, step, /*enable_view_slice=*/false));
    std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>();
    if (as_tuple) {
      const auto& transposed_output = JUST(functional::Transpose2dim(output, 1, 0));
      for (int64_t i = 0; i < ndim; ++i) {
        outputs->emplace_back(
            JUST(functional::TensorGetItem(transposed_output, {functional::detail::IndexItem(i)})));
      }
    } else {
      outputs->emplace_back(output);
    }
    return outputs;
  }
};

class BroadcastLikeFunctor {
 public:
  BroadcastLikeFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("broadcast_like").Input("x").Input("like").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& like,
                           const std::vector<int32_t>& broadcast_axes) const {
    const Shape& x_shape = *x->shape();
    const Shape& like_shape = *like->shape();
    if (x_shape == like_shape) { return x; }
    CHECK_GE_OR_RETURN(like_shape.NumAxes(), x_shape.NumAxes())
        << Error::RuntimeError() << "The number of sizes provided (" << like_shape.NumAxes()
        << ") must be greater or equal to the number of dimensions in the tensor ("
        << x_shape.NumAxes() << ")"
        << ". Target sizes: " << like_shape.ToString() << ". Tensor sizes: " << x_shape.ToString();
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("broadcast_axes");
    if (broadcast_axes.empty()) {
      int64_t like_ndim = like_shape.NumAxes();
      int64_t x_ndim = x_shape.NumAxes();
      int64_t num_prepend = like_ndim - x_ndim;
      std::vector<int64_t> prepend_shape(num_prepend, 1);
      std::vector<int32_t> broadcast_axes;
      for (int i = 0; i < x_ndim; ++i) { prepend_shape.emplace_back(x_shape.At(i)); }
      for (int i = 0; i < num_prepend; ++i) { broadcast_axes.emplace_back(i); }
      for (int i = num_prepend; i < prepend_shape.size(); ++i) {
        if (prepend_shape[i] != like_shape.At(i)) {
          if (prepend_shape[i] == 1) {
            broadcast_axes.emplace_back(i);
          } else {
            return Error::RuntimeError() << "The expanded size of the tensor "
                                         << "(" << like_shape.At(i) << ")"
                                         << " must match the existing size (" << prepend_shape[i]
                                         << ") at non-singleton dimension " << i
                                         << ". Target sizes: " << like_shape.ToString()
                                         << ". Tensor sizes: " << x_shape.ToString();
          }
        }
      }
      attrs.SetAllAttrs(broadcast_axes);
    } else {
      attrs.SetAllAttrs(broadcast_axes);
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, JUST(like->detach())}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ConcatFunctor {
 public:
  ConcatFunctor() {
    ops_.resize(kMaxInputCount);
    for (int n = 0; n < ops_.size(); ++n) {
      ops_[n] = CHECK_JUST(one::OpBuilder("concat").Input("in", n + 1).Output("out").Build());
    }
  }
  Maybe<Tensor> operator()(const TensorTuple& inputs, const int64_t& dim) const {
    const int64_t ninput = inputs.size();
    int64_t axis = dim;
    int64_t ndim = inputs[0]->ndim();
    int64_t nelement = inputs[0]->nelement();
    int64_t max_dim_size = 0;
    CHECK_GE_OR_RETURN(ninput, 1) << Error::RuntimeError() << "inputs size must greater than 0";
    axis = JUST(maybe_wrap_dim(axis, ndim));

    const std::shared_ptr<const Shape>& shape = inputs[0]->shape();
    for (const auto& input : inputs) {
      if (nelement == 0 and ndim == 1) {
        if (input->nelement() != 0 or input->ndim() != 1) {
          ndim = input->ndim();
          nelement = input->nelement();
        } else {
          continue;
        }
      } else if (input->nelement() != 0 or input->ndim() != 1) {
        CHECK_OR_RETURN(input->ndim() == ndim)
            << Error::RuntimeError() << "Tensors must have same number of dimensions: got " << ndim
            << " and " << input->ndim() << " is expected.";
      }
      for (int i = 0; i < ndim; ++i) {
        if (input->nelement() == 0 and input->ndim() == 1) { continue; }
        if (axis == i) {
          max_dim_size += input->shape()->At(i);
        } else if (inputs[0]->nelement() != 0) {
          CHECK_OR_RETURN(input->shape()->At(i) == shape->At(i))
              << Error::RuntimeError() << "Sizes of tensors must match except in dimension " << axis
              << ". Got " << input->shape()->At(i) << " and " << shape->At(i)
              << " is expected in dimension " << i << ".";
        }
      }
    }

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("axis", "max_dim_size");
    attrs.SetAllAttrs(axis, max_dim_size);
    TensorTuple outputs;
    for (int i = 0; i < ninput; i += kMaxInputCount) {
      size_t size = (i + kMaxInputCount) < ninput ? kMaxInputCount : ninput - i;
      TensorTuple partial_inputs(size);
      TensorProcessor tensor_processor;
      for (int j = 0; j < size; ++j) { partial_inputs[j] = inputs[i + j]; }
      JUST(tensor_processor.PromoteInputsToCommonDtype(true)
               .AddInputs(partial_inputs, inputs.at(i)->dtype())
               .Apply());
      TensorTuple input_tuple = JUST(tensor_processor.GetInputs());
      outputs.emplace_back(
          JUST(OpInterpUtil::Dispatch<Tensor>(*ops_[size - 1], input_tuple, attrs)));
    }

    if (outputs.size() == 1) { return outputs.at(0); }
    return this->operator()(outputs, axis);
  }

 private:
  std::vector<std::shared_ptr<OpExpr>> ops_;
};

class StackFunctor {
 public:
  StackFunctor() {
    ops_.resize(kMaxInputCount);
    for (int n = 0; n < ops_.size(); ++n) {
      ops_[n] = CHECK_JUST(one::OpBuilder("stack").Input("in", n + 1).Output("out").Build());
    }
  }
  Maybe<Tensor> operator()(const TensorTuple& inputs, const int64_t& dim) const {
    const int64_t ninput = inputs.size();
    int64_t ndims = inputs[0]->ndim();
    int64_t stack_dim = dim;
    stack_dim = JUST(maybe_wrap_dim(stack_dim, ndims + 1));
    const std::shared_ptr<const Shape>& first_in_shape = inputs[0]->shape();
    for (const auto& input : inputs) {
      for (int i = 0; i < ndims; ++i) {
        CHECK_OR_RETURN(input->shape()->At(i) == first_in_shape->At(i))
            << Error::RuntimeError() << "stack expects each tensor to be equal size, but got "
            << first_in_shape->ToString() << " at first input and " << input->shape()->ToString()
            << " which index is " << i;
      }
    }
    int64_t max_dim_size = ninput;
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("axis", "max_dim_size");
    attrs.SetAllAttrs(stack_dim, max_dim_size);
    TensorTuple outputs;
    for (int i = 0; i < ninput; i += kMaxInputCount) {
      size_t size = (i + kMaxInputCount) < ninput ? kMaxInputCount : ninput - i;
      TensorTuple partial_inputs(size);
      for (int j = 0; j < size; ++j) { partial_inputs[j] = inputs[i + j]; }
      if (partial_inputs.size() == 1) {
        // Use ExpandDims functor for only one input
        outputs.emplace_back(JUST(functional::ExpandDims(partial_inputs[0], dim)));
      } else {
        outputs.emplace_back(
            JUST(OpInterpUtil::Dispatch<Tensor>(*ops_[size - 1], partial_inputs, attrs)));
      }
    }
    if (outputs.size() == 1) { return outputs.at(0); }
    return Concat(outputs, stack_dim);
  }

 private:
  std::vector<std::shared_ptr<OpExpr>> ops_;
};

class StackGradFunctor {
 public:
  StackGradFunctor() {
    ops_.resize(kMaxInputCount);
    for (int n = 1; n < ops_.size(); ++n) {
      ops_[n] = CHECK_JUST(one::OpBuilder("stack_grad")
                               .Input("in")
                               .Input("like", n + 1)
                               .Output("out", n + 1)
                               .Build());
    }
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& x, const TensorTuple& like,
                                const int64_t& axis) const {
    CHECK_GE_OR_RETURN(like.size(), 2)
        << Error::RuntimeError() << "like.size() must not less than 2, but got " << like.size();
    CHECK_LE_OR_RETURN(like.size(), kMaxInputCount)
        << Error::RuntimeError() << "like.size() must not greater than " << kMaxInputCount
        << ", but got " << like.size();
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("axis");
    attrs.SetAllAttrs(axis);
    TensorTuple inputs(like.size() + 1);
    inputs[0] = x;
    for (int i = 0; i < like.size(); ++i) { inputs[i + 1] = like[i]; }
    return OpInterpUtil::Dispatch<TensorTuple>(*ops_.at(like.size() - 1), inputs, attrs);
  }

 private:
  std::vector<std::shared_ptr<OpExpr>> ops_;
};

class AtLeast1DFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x) const {
    if (x->ndim() == 0) {
      return JUST(Reshape(x, {1}));
    } else
      return x;
  }
};

class AtLeast1DListFunctor {
 public:
  Maybe<TensorTuple> operator()(const TensorTuple& inputs) const {
    TensorTuple result = TensorTuple(inputs.size());
    for (int32_t i = 0; i < inputs.size(); i++) {
      result.at(i) = JUST(AtLeast1D(JUST(VectorAt(inputs, i))));
    }
    return result;
  }
};

class AtLeast2DFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x) const {
    if (x->ndim() == 0) {
      return JUST(Reshape(x, {1, 1}));
    } else if (x->ndim() == 1) {
      return JUST(Unsqueeze(x, 0));
    } else
      return x;
  }
};

class AtLeast2DListFunctor {
 public:
  Maybe<TensorTuple> operator()(const TensorTuple& inputs) const {
    TensorTuple result = TensorTuple(inputs.size());
    for (int32_t i = 0; i < inputs.size(); i++) {
      result.at(i) = JUST(AtLeast2D(JUST(VectorAt(inputs, i))));
    }
    return result;
  }
};

class AtLeast3DFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x) const {
    if (x->ndim() == 0) {
      return JUST(Reshape(x, {1, 1, 1}));
    } else if (x->ndim() == 1) {
      return JUST(Reshape(x, {1, x->shape()->At(0), 1}));
    } else if (x->ndim() == 2) {
      return JUST(Unsqueeze(x, -1));
    } else
      return x;
  }
};

class AtLeast3DListFunctor {
 public:
  Maybe<TensorTuple> operator()(const TensorTuple& inputs) const {
    TensorTuple result = TensorTuple(inputs.size());
    for (int32_t i = 0; i < inputs.size(); i++) {
      result.at(i) = JUST(AtLeast3D(JUST(VectorAt(inputs, i))));
    }
    return result;
  }
};

class ColumnStackFunctor {
 public:
  Maybe<Tensor> operator()(const TensorTuple& inputs) const {
    std::shared_ptr<TensorTuple> new_inputs = std::make_shared<TensorTuple>(inputs.size());
    for (int32_t i = 0; i < inputs.size(); i++) {
      const auto& t = JUST(VectorAt(inputs, i));
      if (t->ndim() <= 1)
        new_inputs->at(i) = JUST(Reshape(t, {t->nelement(), 1}));
      else
        new_inputs->at(i) = t;
    }
    return HStack(*new_inputs);
  }
};

class HStackFunctor {
 public:
  Maybe<Tensor> operator()(const TensorTuple& inputs) const {
    std::shared_ptr<TensorTuple> new_inputs = JUST(AtLeast1D(inputs));
    if (new_inputs->at(0)->ndim() == 1)
      return Concat(*new_inputs, 0);
    else
      return Concat(*new_inputs, 1);
  }
};

class VStackFunctor {
 public:
  Maybe<Tensor> operator()(const TensorTuple& inputs) const {
    std::shared_ptr<TensorTuple> new_inputs = JUST(AtLeast2D(inputs));
    return Concat(*new_inputs, 0);
  }
};

class RowStackFunctor {
 public:
  Maybe<Tensor> operator()(const TensorTuple& inputs) const { return VStack(inputs); }
};

class DStackFunctor {
 public:
  Maybe<Tensor> operator()(const TensorTuple& inputs) const {
    std::shared_ptr<TensorTuple> new_inputs = JUST(AtLeast3D(inputs));
    return Concat(*new_inputs, 2);
  }
};

class ExpandFunctor {
 public:
  ExpandFunctor() { op_ = CHECK_JUST(one::OpBuilder("expand").Input("in").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Shape& shape) const {
    const Shape& in_shape = *x->shape();
    int lpad = shape.size() - in_shape.size();
    if (lpad < 0) {
      return Error::RuntimeError()
             << "expand(tensor{" << in_shape.ToString() << "}, size=" << in_shape.size()
             << "): the number of sizes provided (" << shape.size() << ") "
             << "must be greater or equal to the number of dimensions in the tensor ("
             << in_shape.size() << ")";
    }

    DimVector expand_shape_vec = shape.dim_vec();
    for (size_t i = 0; i < shape.size(); ++i) {
      const auto& t_dim = shape[i];
      if (t_dim < -1) {
        return Error::RuntimeError() << "Trying to create tensor with negative dimension " << t_dim;
      }
      if (i >= lpad) {
        const auto& dim = in_shape[i - lpad];
        if (dim != 1 && t_dim != -1 && t_dim != dim) {
          return Error::RuntimeError()
                 << "The expanded size of the tensor (" << t_dim
                 << ") must match the existing size (" << dim << ") at non-singleton dimension "
                 << i << ". Target sizes: " << shape.ToString()
                 << ". Tensor sizes: " << in_shape.ToString();
        }
        if (t_dim == -1) { expand_shape_vec[i] = dim; }
      } else {
        if (t_dim == -1) {
          return Error::RuntimeError() << "The expanded size of the tensor (-1) isn't allowed in a "
                                          "leading, non-existing dimension "
                                       << i;
        }
      }
    }

    // if input tensor is eager local, then try return tensor's view
    Shape expand_shape(expand_shape_vec);
    if (view::IsViewApplicable(x)) { return view::Expand(x, expand_shape); }

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("expand_shape");
    attrs.SetAllAttrs(expand_shape);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ExpandDimsFunctor {
 public:
  ExpandDimsFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("expand_dims").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int32_t& dim) const {
    int32_t expand_dim = dim;
    const int32_t ndim = input->shape()->NumAxes();
    expand_dim = JUST(maybe_wrap_dim(dim, ndim + 1));

    if (view::IsViewApplicable(input)) { return view::Unsqueeze(input, expand_dim); }

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("axis");
    attrs.SetAllAttrs(expand_dim);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UnsqueezeMultipleFunctor {
 public:
  UnsqueezeMultipleFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& dim,
                           const int32_t& n_dims) const {
    if (dim.size() == 0 || x->ndim() == n_dims) {
      return x;
    } else if (dim.size() == 1) {
      return JUST(functional::Unsqueeze(x, JUST(VectorAt(dim, 0))));
    } else {
      std::shared_ptr<Tensor> tensor = x;
      const auto& dims_to_unsqueeze = JUST(dim_list_to_bitset(dim, n_dims));

      // Unsqueeze is called several times to extend the dimension when the View mechanism is
      // enabled. Otherwise, calculate the target shape and call reshape.
      if (view::IsViewApplicable(tensor)) {
        for (int32_t i = 0; i < n_dims; i++) {
          if ((*dims_to_unsqueeze)[i]) { tensor = JUST(view::Unsqueeze(tensor, i)); }
        }
      } else {
        std::vector<int64_t> target_dims(n_dims, 0);
        int32_t tensor_index = 0;
        for (int32_t i = 0; i < n_dims; i++) {
          if ((*dims_to_unsqueeze)[i]) {
            target_dims[i] = 1;
          } else {
            CHECK_LT_OR_RETURN(tensor_index, tensor->ndim());  // NOLINT(maybe-need-error-msg)
            target_dims[i] = tensor->shape()->at(tensor_index);
            tensor_index++;
          }
        }
        Shape infered_shape(DimVector(target_dims.begin(), target_dims.end()));
        tensor = JUST(functional::Reshape(tensor, infered_shape));
      }
      return tensor;
    }
  }
};

class InplaceUnsqueezeFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int32_t& dim) const {
    JUST(CheckInplaceValid(input));
    const int64_t expand_dim = JUST(maybe_wrap_dim(dim, input->shape()->NumAxes() + 1));
    CHECK_OR_RETURN(view::IsViewApplicable(input))
        << "inplace unsqueeze(tensor.unsqueeze_) only support in eager local mode!";

    JUST(view::InplaceUnsqueeze(input, expand_dim));
    return input;
  }
};

class SqueezeFunctor {
 public:
  SqueezeFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("squeeze").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const Optional<std::vector<int32_t>>& dim) const {
    int32_t ndim = x->shape()->NumAxes();
    std::vector<int32_t> squeeze_dims;
    squeeze_dims.reserve(ndim);
    if (dim.has_value()) {
      std::vector<int32_t> dims = *JUST(dim);
      for (int32_t dim_i : dims) {
        dim_i = JUST(maybe_wrap_dim(dim_i, ndim));
        if (x->shape()->At(dim_i) == 1) { squeeze_dims.emplace_back(dim_i); }
      }
    } else {
      for (int i = 0; i < ndim; ++i) {
        if (x->shape()->At(i) == 1) { squeeze_dims.emplace_back(i); }
      }
    }

    if (view::IsViewApplicable(x)) { return view::Squeeze(x, squeeze_dims); }

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("axes");
    attrs.SetAllAttrs(squeeze_dims);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class InplaceSqueezeFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const Optional<std::vector<int32_t>>& dim) const {
    JUST(CheckInplaceValid(input));
    const int32_t ndim = input->shape()->NumAxes();
    std::vector<int32_t> squeeze_dims;
    squeeze_dims.reserve(ndim);
    if (dim.has_value()) {
      std::vector<int32_t> dims = *JUST(dim);
      for (int32_t dim_i : dims) {
        dim_i = JUST(maybe_wrap_dim(dim_i, ndim));
        if (input->shape()->At(dim_i) == 1) { squeeze_dims.emplace_back(dim_i); }
      }
    } else {
      for (int i = 0; i < ndim; ++i) {
        if (input->shape()->At(i) == 1) { squeeze_dims.emplace_back(i); }
      }
    }

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("axes");
    attrs.SetAllAttrs(squeeze_dims);

    CHECK_OR_RETURN(view::IsViewApplicable(input))
        << "inplace squeeze(tensor.squeeze_) only support in eager local mode!";

    JUST(view::InplaceSqueeze(input, squeeze_dims));
    return input;
  }
};

class RollFunctor {
 public:
  RollFunctor() { op_ = CHECK_JUST(one::OpBuilder("roll").Input("in").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::vector<int32_t>& shifts,
                           const Optional<std::vector<int32_t>>& dims) const {
    std::vector<int32_t> actual_dims;
    if (dims.has_value()) {
      actual_dims = *JUST(dims);
    } else {
      actual_dims.emplace_back(-1);
    }
    CHECK_EQ_OR_RETURN(shifts.size(), actual_dims.size())
        << Error::RuntimeError() << "shifts and dimensions must align. shifts: " << shifts.size()
        << ", dims: " << actual_dims.size();

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("shifts", "dims");
    attrs.SetAllAttrs(shifts, actual_dims);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class GatherFunctor {
 public:
  GatherFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("gather").Input("in").Input("indices").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& indices, const int64_t& axis) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("axis");
    attrs.SetAllAttrs(axis);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, indices}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DimGatherFunctor {
 public:
  DimGatherFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("dim_gather").Input("input").Input("index").Output("output").Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int64_t& dim,
                           const std::shared_ptr<one::Tensor>& index,
                           const bool sparse_grad) const {
    CHECK_OR_RETURN(index->dtype()->data_type() == kInt64 || index->dtype()->data_type() == kInt32)
        << Error::RuntimeError() << "gather(): Expected dtype int32 or int64 for index";
    CHECK_EQ_OR_RETURN(sparse_grad, false)
        << Error::RuntimeError() << "Only support bool = False for now!";

    int64_t new_dim = JUST(maybe_wrap_dim(dim, index->ndim()));
    if (input->ndim() > 0 && index->ndim() > 0) {
      CHECK_EQ_OR_RETURN(input->ndim(), index->ndim())
          << Error::RuntimeError()
          << "Index tensor must have the same number of dimensions as input tensor";
    } else if (input->ndim() == 0) {
      CHECK_LE_OR_RETURN(index->ndim(), 1)
          << Error::RuntimeError()
          << "Index tensor must have the same number of dimensions as input tensor";
    } else {
      CHECK_LE_OR_RETURN(input->ndim(), 1)
          << Error::RuntimeError()
          << "Index tensor must have the same number of dimensions as input tensor";
    }
    if (input->ndim() > 0 && index->ndim() > 0) {
      FOR_RANGE(int32_t, i, 0, input->ndim()) {
        if (i != new_dim) {
          CHECK_LE_OR_RETURN(index->shape()->At(i), input->shape()->At(i))
              << Error::RuntimeError() << "Size does not match at dimension " << i
              << " expected index " << *(index->shape()) << " to be smaller than self "
              << *(input->shape()) << " apart from dimension " << new_dim;
        }
      }
    }

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("dim");
    attrs.SetAllAttrs(static_cast<int32_t>(new_dim));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input, index}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

enum class DimScatterType { kUpdate, kAdd, kMultiply };

template<DimScatterType T>
std::string DimScatterTypeToString() {
  switch (T) {
    case DimScatterType::kUpdate: return "_update";
    case DimScatterType::kAdd: return "_add";
    case DimScatterType::kMultiply: return "_mul";
  }
  return "";
}

template<DimScatterType T>
class DimScatterFunctorImpl {
 public:
  DimScatterFunctorImpl()
      : op_(CHECK_JUST(one::OpBuilder("dim_scatter" + DimScatterTypeToString<T>())
                           .Input("input")
                           .Input("index")
                           .Input("src")
                           .Output("output")
                           .Build())) {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int32_t& dim,
                           const std::shared_ptr<one::Tensor>& index,
                           const std::shared_ptr<one::Tensor>& src, bool inplace) const {
    const int32_t ndim = input->shape()->NumAxes();
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("dim");
    attrs.SetAllAttrs(static_cast<int32_t>(JUST(maybe_wrap_dim(dim, ndim))));
    if (inplace) {
      JUST(CheckInplaceValid(input));
      auto outputs = std::make_shared<TensorTuple>(1);
      outputs->at(0) = input;
      JUST(OpInterpUtil::Dispatch(*op_, {input, index, src}, outputs.get(), attrs));
      return outputs->at(0);
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input, index, src}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DimScatterFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int32_t& dim,
                           const std::shared_ptr<one::Tensor>& index,
                           const std::shared_ptr<one::Tensor>& src,
                           const Optional<std::string>& reduce, bool inplace) const {
    if (reduce.has_value()) {
      const std::string& reduce_str = *JUST(reduce);
      if (reduce_str == "add") {
        return DimScatterAdd(input, dim, index, src, inplace);
      } else if (reduce_str == "multiply") {
        return DimScatterMul(input, dim, index, src, inplace);
      } else {
        CHECK_OR_RETURN(false) << Error::RuntimeError() << "Invalid reduce type: " << reduce_str;
      }
    }
    return functional::DimScatterUpdate(input, dim, index, src, inplace);
  }
};

template<DimScatterType T>
class DimScatterScalarFunctorImpl {
 public:
  DimScatterScalarFunctorImpl()
      : op_(CHECK_JUST(one::OpBuilder("dim_scatter" + DimScatterTypeToString<T>() + "_scalar")
                           .Input("input")
                           .Input("index")
                           .Output("output")
                           .Build())) {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int32_t& dim,
                           const std::shared_ptr<one::Tensor>& index, const Scalar& src,
                           bool inplace) const {
    const int32_t ndim = input->shape()->NumAxes();
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("dim", "src_scalar");
    attrs.SetAllAttrs(static_cast<int32_t>(JUST(maybe_wrap_dim(dim, ndim))), src.As<float>());
    if (inplace) {
      JUST(CheckInplaceValid(input));
      auto outputs = std::make_shared<TensorTuple>(1);
      outputs->at(0) = input;
      JUST(OpInterpUtil::Dispatch(*op_, {input, index}, outputs.get(), attrs));
      return outputs->at(0);
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input, index}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DimScatterScalarFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int32_t& dim,
                           const std::shared_ptr<one::Tensor>& index, const Scalar& src,
                           const Optional<std::string>& reduce, bool inplace) const {
    if (reduce.has_value()) {
      const std::string& reduce_str = *JUST(reduce);
      if (reduce_str == "add") {
        return DimScatterAddScalar(input, dim, index, src, inplace);
      } else if (reduce_str == "multiply") {
        return DimScatterMulScalar(input, dim, index, src, inplace);
      } else {
        CHECK_OR_RETURN(false) << Error::RuntimeError() << "Invalid reduce type: " << reduce_str;
      }
    }
    return functional::DimScatterUpdateScalar(input, dim, index, src, inplace);
  }
};

class DimScatterAddLikeFunctor {
 public:
  DimScatterAddLikeFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("dim_scatter_add_like")
                         .Input("like")
                         .Input("index")
                         .Input("src")
                         .Output("output")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& like, const int32_t& dim,
                           const std::shared_ptr<one::Tensor>& index,
                           const std::shared_ptr<one::Tensor>& src) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("dim");
    attrs.SetAllAttrs(dim);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {like, index, src}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ArgSortFunctor {
 public:
  ArgSortFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("arg_sort").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& in,
                           const std::string& direction) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("direction");
    attrs.SetAllAttrs(direction);
    CHECK_OR_RETURN(direction == "ASCENDING" || direction == "DESCENDING")
        << Error::RuntimeError()
        << "expected the input direction parameter value is \"ASCENDING\" or \"DESCENDING\", "
        << "but found the value is "
        << "\"" << direction << "\"";
    return OpInterpUtil::Dispatch<Tensor>(*op_, {in}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class SearchSortedFunctor {
 public:
  SearchSortedFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("searchsorted")
                         .Input("sorted_sequence")
                         .Input("values")
                         .Output("out")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& sorted_sequence,
                           const std::shared_ptr<one::Tensor>& values, bool out_int32,
                           bool right) const {
    // checks
    CHECK_OR_RETURN(values->shape()->NumAxes() > 0)
        << "for searchsorted op, input values tensor should have positive dimension";
    CHECK_OR_RETURN(sorted_sequence->shape()->NumAxes() > 0)
        << "for searchsorted op, input sorted_sequence should have positive dimension, "
        << "but got 0 dimension";
    CHECK_OR_RETURN(sorted_sequence->shape()->NumAxes() == 1
                    || sorted_sequence->shape()->MatchBeforeLastDim(*(values->shape())))
        << "for searchsorted op, sorted_sequence should be 1 dimension or the first N-1 dimensions "
        << "of boundaries tensor and input value tensor must match";
    if (out_int32) {
      CHECK_OR_RETURN(sorted_sequence->shape()->At(sorted_sequence->shape()->NumAxes() - 1)
                      < INT32_MAX)
          << "for searchsorted op, the size of input sorted_sequence' last dimension should "
          << "be less than " << INT32_MAX;
    }
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("out_int32", "right");
    attrs.SetAllAttrs(out_int32, right);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {sorted_sequence, values}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class SearchSortedScalarFunctor {
 public:
  SearchSortedScalarFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("searchsorted_scalar").Input("sorted_sequence").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& sorted_sequence,
                           const Scalar& values, bool out_int32, bool right) const {
    // checks
    CHECK_OR_RETURN(sorted_sequence->shape()->NumAxes() == 1)
        << "for searchsorted op, input value can be a scalar only when sorted_sequence tensor "
        << "dimension is 1, but we got sorted_sequence dim(" << sorted_sequence->shape()->NumAxes()
        << ")";
    if (out_int32) {
      CHECK_OR_RETURN(sorted_sequence->shape()->At(sorted_sequence->shape()->NumAxes() - 1)
                      < INT32_MAX)
          << "for searchsorted op, the size of input sorted_sequence' last dimension should "
          << "be less than " << INT32_MAX;
    }
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("out_int32", "right", "values");
    attrs.SetAllAttrs(out_int32, right, values.As<double>());
    return OpInterpUtil::Dispatch<Tensor>(*op_, {sorted_sequence}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class GatherNdFunctor {
 public:
  GatherNdFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("gather_nd").Input("params").Input("indices").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& params,
                           const std::shared_ptr<one::Tensor>& indices) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {params, indices});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ScatterNdFunctor {
 public:
  ScatterNdFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("scatter_nd").Input("indices").Input("updates").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& indices,
                           const std::shared_ptr<one::Tensor>& updates, const Shape& shape) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("shape");
    attrs.SetAllAttrs(shape);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {indices, updates}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class TensorScatterNdUpdateFunctor {
 public:
  TensorScatterNdUpdateFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("tensor_scatter_nd_update")
                         .Input("params")
                         .Input("indices")
                         .Input("updates")
                         .Output("out")
                         .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& tensor,
                           const std::shared_ptr<one::Tensor>& indices,
                           const std::shared_ptr<one::Tensor>& updates, bool inplace) const {
    CHECK_OR_RETURN(*tensor->dtype() == *updates->dtype())
        << Error::RuntimeError() << "The dtype of tensor and updates must be same.";
    std::shared_ptr<Tensor> contiguous_index = JUST(functional::ToContiguous(indices));
    if (inplace) {
      if (tensor->is_global()) {
        // NOTE: global tensor_scatter_nd_update inplace must calculate on another tensor and assign
        // back because of input's sbp limited
        auto output =
            JUST(OpInterpUtil::Dispatch<Tensor>(*op_, {tensor, contiguous_index, updates}));
        int64_t ndim = tensor->shape()->NumAxes();
        // TODO: use inplace copy op to write back to origin tensor
        std::vector<int64_t> start(ndim, 0);
        std::vector<int64_t> stop(tensor->shape()->begin(), tensor->shape()->end());
        std::vector<int64_t> step(ndim, 1);
        return functional::SliceUpdate(tensor, output, start, stop, step, /*inplace=*/true);
      } else {
        JUST(CheckInplaceValid(tensor));
        auto outputs = std::make_shared<TensorTuple>(1);
        (*outputs)[0] = tensor;
        JUST(OpInterpUtil::Dispatch(*op_, {tensor, contiguous_index, updates}, outputs.get()));
        return (*outputs)[0];
      }
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*op_, {tensor, contiguous_index, updates});
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ScatterNdLikeFunctor {
 public:
  ScatterNdLikeFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("scatter_nd_like")
                         .Input("like")
                         .Input("updates")
                         .Input("indices")
                         .Output("out")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& like,
                           const std::shared_ptr<one::Tensor>& updates,
                           const std::shared_ptr<one::Tensor>& indices) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {like, updates, indices});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ReshapeFunctor {
 public:
  ReshapeFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("reshape").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Shape& shape) const {
    Shape infered_shape = *JUST(InferShapeUnspecifiedDim(x->shape()->Count(0), shape));

    if (view::IsViewApplicable(x)) {
      Optional<Stride> infered_stride =
          ComputeStride(*(x->shape()), *JUST(x->stride()), infered_shape);
      if (infered_stride.has_value()) {
        return view::Reshape(x, infered_shape, *JUST(infered_stride));
      }
    }
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("shape");
    attrs.SetAllAttrs(infered_shape);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ViewFunctor {
 public:
  ViewFunctor() { op_ = CHECK_JUST(one::OpBuilder("reshape").Input("in").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Shape& shape) const {
    Shape infered_shape = *JUST(InferShapeUnspecifiedDim(x->shape()->Count(0), shape));
    if (view::IsViewApplicable(x)) {
      Optional<Stride> infered_stride =
          ComputeStride(*(x->shape()), *JUST(x->stride()), infered_shape);
      CHECK_OR_RETURN_ERROR(infered_stride.has_value())
          << Error::RuntimeError()
          << "view size is not compatible with input tensor's size and stride (at least one "
             "dimension spans across two contiguous subspaces). Use .reshape(...) instead.";
      return view::Reshape(x, infered_shape, *JUST(infered_stride));
    }
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("shape");
    attrs.SetAllAttrs(infered_shape);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ToContiguousFunctor {
 public:
  ToContiguousFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("to_contiguous").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input) const {
    if (input->is_global() || input->is_lazy()) { return input; }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class InplaceToContiguousFunctor {
 public:
  InplaceToContiguousFunctor() {
    assign_op_ = CHECK_JUST(one::OpBuilder("assign").Input("ref").Input("value").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input) const {
    // TODO: use original "inplace_to_contiguous" op replace assign
    if (input->is_contiguous()) { return input; }

    auto contiguous_tensor = JUST(functional::ToContiguous(input));
    CHECK_OR_RETURN(input->is_local() && contiguous_tensor->is_local())
        << "Both ref and value must be local tensor.";
    const Stride stride(*input->shape());
    // update stride
    const auto& blob_object = JUST(input->eager_blob_object());
    Symbol<LocalTensorMeta> old_tensor_meta = JUST(input->local_tensor_meta());

    Symbol<LocalTensorMeta> new_tensor_meta = SymbolOf(LocalTensorMeta(
        old_tensor_meta->shape(), stride, old_tensor_meta->dtype(), old_tensor_meta->device()));

    std::shared_ptr<EagerLocalTensorImpl> final_tensor_impl =
        std::make_shared<EagerLocalTensorImpl>(JUST(input->tensor_storage()),
                                               JUST(input->storage_offset()),
                                               input->requires_grad(), input->is_leaf());
    JUST(final_tensor_impl->set_retain_grad(input->retain_grad()));
    JUST(final_tensor_impl->InitEagerBlobObject(new_tensor_meta,
                                                JUST(blob_object->compute_local_dep_object())));
    JUST(JUST(input->AsLocalTensor())->set_impl(final_tensor_impl));

    // assign contiguous tensor data
    JUST(OpInterpUtil::Dispatch<TensorTuple>(*assign_op_, {input, contiguous_tensor}));
    return input;
  }

 private:
  std::shared_ptr<OpExpr> assign_op_;
};

class NarrowFunctor {
 public:
  NarrowFunctor() { op_ = CHECK_JUST(one::OpBuilder("narrow").Input("in").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int64_t& dim,
                           const int64_t& start, const int64_t& length) const {
    int64_t narrow_dim = dim;
    int64_t narrow_start = start;
    const int64_t ndim = input->shape()->NumAxes();
    CHECK_GT_OR_RETURN(ndim, 0) << Error::RuntimeError()
                                << "narrow() cannot be applied to a 0-dim tensor.";
    narrow_dim = JUST(maybe_wrap_dim(narrow_dim, ndim));
    int64_t dim_length = input->shape()->At(narrow_dim);
    CHECK_OR_RETURN((-dim_length <= start) && (start <= dim_length))
        << Error::IndexError() << "Dimension out of range (expected to be in range of [" << -ndim
        << ", " << ndim << "], but got " << start << ")";
    if (narrow_start < 0) { narrow_start += ndim; }
    CHECK_GE_OR_RETURN(dim_length, narrow_start + length)
        << Error::RuntimeError() << "start (" << narrow_start << ") + length (" << length
        << ") exceeds dimension size (" << dim_length << ")";

    if (view::IsViewApplicable(input)) {
      return JUST(view::Narrow(input, narrow_dim, narrow_start, length));
    }
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("dim", "start", "length");
    attrs.SetAllAttrs(narrow_dim, start, length);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class NarrowGradFunctor {
 public:
  NarrowGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("narrow_grad").Input("dy").Input("like").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& like, const int64_t& dim,
                           const int64_t& start, const int64_t& length) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("dim", "start", "length");
    attrs.SetAllAttrs(dim, start, length);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, like}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class SliceFunctor {
 public:
  SliceFunctor() { op_ = CHECK_JUST(one::OpBuilder("slice").Input("x").Output("y").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const std::vector<int64_t>& start,
                           const std::vector<int64_t>& stop, const std::vector<int64_t>& step,
                           const Optional<bool>& enable_view_slice) const {
    if (view::IsViewApplicable(x) && enable_view_slice.value_or(false)) {
      return view::Slice(x, start, stop, step);
    }

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("start", "stop", "step");
    attrs.SetAllAttrs(start, stop, step);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 protected:
  std::shared_ptr<OpExpr> op_;
};

class SliceUpdateFunctor {
 public:
  SliceUpdateFunctor() {
    op_ =
        CHECK_JUST(one::OpBuilder("slice_update").Input("ref").Input("value").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& ref,
                           const std::shared_ptr<one::Tensor>& value,
                           const std::vector<int64_t>& start, const std::vector<int64_t>& stop,
                           const std::vector<int64_t>& step, bool inplace) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("start", "stop", "step");
    attrs.SetAllAttrs(start, stop, step);

    TensorProcessor tensor_processor;
    JUST(tensor_processor.AddInputs({ref, value})
             .PromoteInputsToCommonDtype(true, ref->dtype())
             .Apply());

    if (inplace) {
      auto outputs = std::make_shared<TensorTuple>(1);
      JUST(CheckInplaceValid(ref));
      JUST(VectorAt(*outputs, 0)) = ref;
      JUST(OpInterpUtil::Dispatch(*op_, JUST(tensor_processor.GetInputs()), outputs.get(), attrs));
      return JUST(VectorAt(*outputs, 0));
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*op_, JUST(tensor_processor.GetInputs()), attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class SliceGradFunctor {
 public:
  SliceGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("slice_grad").Input("dy").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy, const Shape& like_shape,
                           const std::vector<int64_t>& start, const std::vector<int64_t>& stop,
                           const std::vector<int64_t>& step) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("like_shape", "start", "stop", "step");
    attrs.SetAllAttrs(like_shape, start, stop, step);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy}, attrs);
  }

 protected:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleGradFunctor {
 public:
  UpsampleGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("upsample_grad").Input("dy").Input("x").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x, const double& height_scale,
                           const double& width_scale, const bool& align_corners,
                           const std::string& data_format, const std::string& interpolation) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("height_scale", "width_scale", "align_corners",
                                                 "interpolation", "data_format");
    attrs.SetAllAttrs(height_scale, width_scale, align_corners, interpolation, data_format);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class CopyToDeviceFunctor {
 public:
  CopyToDeviceFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("copy").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, Symbol<Device> device,
                           const bool pin_memory) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("device", "pin_memory");
    attrs.SetAllAttrs(device, pin_memory);

#ifdef WITH_CUDA
    if (device->enum_type() == DeviceType::kCUDA) { InitCudaContextOnce(device->device_id()); }
#endif
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class CopyFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const std::string& device_type,
                           const int64_t& device_id, const bool pin_memory) const {
    return functional::Copy(x, JUST(Device::New(device_type, device_id)), pin_memory);
  }
};

class FlipFunctor {
 public:
  FlipFunctor() { op_ = CHECK_JUST(one::OpBuilder("flip").Input("x").Output("y").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::vector<int32_t>& dims) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("dims");
    if (dims.empty()) {
      attrs.SetAllAttrs(dims);
    } else {
      std::vector<int32_t> flip_dims = *JUST(CheckAxis(dims, x->ndim()));
      attrs.SetAllAttrs(flip_dims);
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UnfoldTensorFunctor {
 public:
  UnfoldTensorFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("unfold_tensor").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const int32_t& dimension,
                           const int32_t& size, const int32_t& step) const {
    // if input tensor is eager local, than try return tensor's view
    if (view::IsViewApplicable(x)) { return view::UnfoldTensor(x, dimension, size, step); }

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("dimension", "size", "step");
    attrs.SetAllAttrs(dimension, size, step);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UnfoldTensorGradFunctor {
 public:
  UnfoldTensorGradFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("unfold_tensor_grad").Input("dy").Input("x").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x, const int32_t& dimension,
                           const int32_t& size, const int32_t& step) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("dimension", "size", "step");
    attrs.SetAllAttrs(dimension, size, step);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleLinear1DFunctor {
 public:
  UpsampleLinear1DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("upsample_linear_1d").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const double& scale_factor,
                           const bool& align_corners,
                           const Optional<std::vector<int64_t>>& output_size,
                           const std::string& data_format) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("scale_factor", "align_corners", "data_format",
                                                 "output_size");
    if (output_size.has_value()) {
      attrs.SetAllAttrs(scale_factor, align_corners, data_format, *JUST(output_size));
    } else {
      attrs.SetAllAttrs(scale_factor, align_corners, data_format, NullOpt);
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleLinear1DGradFunctor {
 public:
  UpsampleLinear1DGradFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("upsample_linear_1d_grad").Input("dy").Input("x").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x, const double& scale_factor,
                           const bool& align_corners,
                           const Optional<std::vector<int64_t>>& output_size,
                           const std::string& data_format) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("scale_factor", "align_corners", "data_format",
                                                 "output_size");
    if (output_size.has_value()) {
      attrs.SetAllAttrs(scale_factor, align_corners, data_format, *JUST(output_size));
    } else {
      attrs.SetAllAttrs(scale_factor, align_corners, data_format, NullOpt);
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleNearest1DFunctor {
 public:
  UpsampleNearest1DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("upsample_nearest_1d").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const double& scale_factor,
                           const Optional<std::vector<int64_t>>& output_size,
                           const std::string& data_format) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("scale_factor", "data_format", "output_size");
    if (output_size) {
      attrs.SetAllAttrs(scale_factor, data_format, *JUST(output_size));
    } else {
      attrs.SetAllAttrs(scale_factor, data_format, NullOpt);
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleNearest1DGradFunctor {
 public:
  UpsampleNearest1DGradFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("upsample_nearest_1d_grad").Input("dy").Input("x").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x, const double& scale_factor,
                           const Optional<std::vector<int64_t>>& output_size,
                           const std::string& data_format) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("scale_factor", "data_format", "output_size");
    if (output_size) {
      attrs.SetAllAttrs(scale_factor, data_format, *JUST(output_size));
    } else {
      attrs.SetAllAttrs(scale_factor, data_format, NullOpt);
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleNearest2DFunctor {
 public:
  UpsampleNearest2DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("upsample_nearest_2d").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const double& height_scale,
                           const double& width_scale,
                           const Optional<std::vector<int64_t>>& output_size,
                           const std::string& data_format) const {
    auto& attrs =
        THREAD_CACHED_MUTABLE_ATTR_MAP("height_scale", "width_scale", "data_format", "output_size");
    if (output_size) {
      attrs.SetAllAttrs(height_scale, width_scale, data_format, *JUST(output_size));
    } else {
      attrs.SetAllAttrs(height_scale, width_scale, data_format, NullOpt);
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleNearest2DGradFunctor {
 public:
  UpsampleNearest2DGradFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("upsample_nearest_2d_grad").Input("dy").Input("x").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x, const double& height_scale,
                           const double& width_scale,
                           const Optional<std::vector<int64_t>>& output_size,
                           const std::string& data_format) const {
    auto& attrs =
        THREAD_CACHED_MUTABLE_ATTR_MAP("height_scale", "width_scale", "data_format", "output_size");
    if (output_size) {
      attrs.SetAllAttrs(height_scale, width_scale, data_format, *JUST(output_size));
    } else {
      attrs.SetAllAttrs(height_scale, width_scale, data_format, NullOpt);
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleBilinear2DFunctor {
 public:
  UpsampleBilinear2DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("upsample_bilinear_2d").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const double& height_scale,
                           const double& width_scale, const bool& align_corners,
                           const Optional<std::vector<int64_t>>& output_size,
                           const std::string& data_format) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("height_scale", "width_scale", "align_corners",
                                                 "data_format", "output_size");
    if (output_size) {
      attrs.SetAllAttrs(height_scale, width_scale, align_corners, data_format, *JUST(output_size));
    } else {
      attrs.SetAllAttrs(height_scale, width_scale, align_corners, data_format, NullOpt);
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleBilinear2DGradFunctor {
 public:
  UpsampleBilinear2DGradFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("upsample_bilinear_2d_grad").Input("dy").Input("x").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x, const double& height_scale,
                           const double& width_scale, const bool& align_corners,
                           const Optional<std::vector<int64_t>>& output_size,
                           const std::string& data_format) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("height_scale", "width_scale", "align_corners",
                                                 "data_format", "output_size");
    if (output_size) {
      attrs.SetAllAttrs(height_scale, width_scale, align_corners, data_format, *JUST(output_size));
    } else {
      attrs.SetAllAttrs(height_scale, width_scale, align_corners, data_format, NullOpt);
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleBicubic2DFunctor {
 public:
  UpsampleBicubic2DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("upsample_bicubic_2d").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const double& height_scale,
                           const double& width_scale, const bool& align_corners,
                           const Optional<std::vector<int64_t>>& output_size,
                           const std::string& data_format) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("height_scale", "width_scale", "align_corners",
                                                 "data_format", "output_size");
    if (output_size) {
      attrs.SetAllAttrs(height_scale, width_scale, align_corners, data_format, *JUST(output_size));
    } else {
      attrs.SetAllAttrs(height_scale, width_scale, align_corners, data_format, NullOpt);
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleBicubic2DGradFunctor {
 public:
  UpsampleBicubic2DGradFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("upsample_bicubic_2d_grad").Input("dy").Input("x").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x, const double& height_scale,
                           const double& width_scale, const bool& align_corners,
                           const Optional<std::vector<int64_t>>& output_size,
                           const std::string& data_format) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("height_scale", "width_scale", "align_corners",
                                                 "data_format", "output_size");
    if (output_size) {
      attrs.SetAllAttrs(height_scale, width_scale, align_corners, data_format, *JUST(output_size));
    } else {
      attrs.SetAllAttrs(height_scale, width_scale, align_corners, data_format, NullOpt);
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleNearest3DFunctor {
 public:
  UpsampleNearest3DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("upsample_nearest_3d").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const double& depth_scale,
                           const double& height_scale, const double& width_scale,
                           const Optional<std::vector<int64_t>>& output_size,
                           const std::string& data_format) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("depth_scale", "height_scale", "width_scale",
                                                 "data_format", "output_size");
    if (output_size) {
      attrs.SetAllAttrs(depth_scale, height_scale, width_scale, data_format, *JUST(output_size));
    } else {
      attrs.SetAllAttrs(depth_scale, height_scale, width_scale, data_format, NullOpt);
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleNearest3DGradFunctor {
 public:
  UpsampleNearest3DGradFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("upsample_nearest_3d_grad").Input("dy").Input("x").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x, const double& depth_scale,
                           const double& height_scale, const double& width_scale,
                           const Optional<std::vector<int64_t>>& output_size,
                           const std::string& data_format) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("depth_scale", "height_scale", "width_scale",
                                                 "data_format", "output_size");
    if (output_size) {
      attrs.SetAllAttrs(depth_scale, height_scale, width_scale, data_format, *JUST(output_size));
    } else {
      attrs.SetAllAttrs(depth_scale, height_scale, width_scale, data_format, NullOpt);
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleTrilinear3DFunctor {
 public:
  UpsampleTrilinear3DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("upsample_trilinear_3d").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const double& depth_scale,
                           const double& height_scale, const double& width_scale,
                           const bool& align_corners,
                           const Optional<std::vector<int64_t>>& output_size,
                           const std::string& data_format) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("depth_scale", "height_scale", "width_scale",
                                                 "align_corners", "data_format", "output_size");
    if (output_size) {
      attrs.SetAllAttrs(depth_scale, height_scale, width_scale, align_corners, data_format,
                        *JUST(output_size));
    } else {
      attrs.SetAllAttrs(depth_scale, height_scale, width_scale, align_corners, data_format,
                        NullOpt);
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleTrilinear3DGradFunctor {
 public:
  UpsampleTrilinear3DGradFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("upsample_trilinear_3d_grad").Input("dy").Input("x").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x, const double& depth_scale,
                           const double& height_scale, const double& width_scale,
                           const bool& align_corners,
                           const Optional<std::vector<int64_t>>& output_size,
                           const std::string& data_format) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("depth_scale", "height_scale", "width_scale",
                                                 "align_corners", "data_format", "output_size");
    if (output_size) {
      attrs.SetAllAttrs(depth_scale, height_scale, width_scale, align_corners, data_format,
                        *JUST(output_size));
    } else {
      attrs.SetAllAttrs(depth_scale, height_scale, width_scale, align_corners, data_format,
                        NullOpt);
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UnsortedSegmentSumLikeFunctor {
 public:
  UnsortedSegmentSumLikeFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("unsorted_segment_sum_like")
                         .Input("data")
                         .Input("segment_ids")
                         .Input("like")
                         .Output("out")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& segment_ids,
                           const std::shared_ptr<one::Tensor>& like, const int64_t& axis) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("axis");
    attrs.SetAllAttrs(axis);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, segment_ids, like}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UnsortedSegmentSumFunctor {
 public:
  UnsortedSegmentSumFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("unsorted_segment_sum")
                         .Input("data")
                         .Input("segment_ids")
                         .Output("out")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& segment_ids, const int64_t& axis,
                           const int64_t& num_segments) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("axis", "num_segments");
    attrs.SetAllAttrs(axis, num_segments);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, segment_ids}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class TrilFunctor {
 public:
  TrilFunctor() { op_ = CHECK_JUST(one::OpBuilder("tril").Input("in").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const int64_t& diagonal) const {
    auto& attrs =
        THREAD_CACHED_MUTABLE_ATTR_MAP("diagonal", "is_floating_fill_value", "integer_fill_value");
    attrs.SetAllAttrs(diagonal, false, static_cast<int64_t>(0));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class InplaceTrilFunctor {
 public:
  InplaceTrilFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("tril").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const int64_t& diagonal) const {
    JUST(CheckInplaceValid(x));
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("diagonal");
    attrs.SetAllAttrs(diagonal);
    std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
    outputs->at(0) = x;
    JUST(OpInterpUtil::Dispatch(*op_, {x}, outputs.get(), attrs));
    return outputs->at(0);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class TriuFunctor {
 public:
  TriuFunctor() { op_ = CHECK_JUST(one::OpBuilder("triu").Input("in").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const int64_t& diagonal) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("diagonal");
    attrs.SetAllAttrs(diagonal);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class InplaceTriuFunctor {
 public:
  InplaceTriuFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("triu").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const int64_t& diagonal) const {
    JUST(CheckInplaceValid(x));
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("diagonal");
    attrs.SetAllAttrs(diagonal);
    std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
    outputs->at(0) = x;
    JUST(OpInterpUtil::Dispatch(*op_, {x}, outputs.get(), attrs));
    return outputs->at(0);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DiagFunctor {
 public:
  DiagFunctor() { op_ = CHECK_JUST(one::OpBuilder("diag").Input("in").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const int32_t& diagonal) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("diagonal");
    attrs.SetAllAttrs(diagonal);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DiagGradFunctor {
 public:
  DiagGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("diag_grad").Input("dy").Input("in").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x, const int32_t& diagonal) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("diagonal");
    attrs.SetAllAttrs(diagonal);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DiagonalFunctor {
 public:
  DiagonalFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("diagonal").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const int32_t& offset,
                           const int32_t& dim1, const int32_t& dim2) const {
    int64_t ndims = x->shape()->NumAxes();
    int32_t p_dim1 = dim1;
    int32_t p_dim2 = dim2;
    p_dim1 = JUST(maybe_wrap_dim(p_dim1, ndims));
    p_dim2 = JUST(maybe_wrap_dim(p_dim2, ndims));
    CHECK_NE_OR_RETURN(p_dim1, p_dim2)
        << Error::RuntimeError() << "diagonal dimensions cannot be identical " << dim1 << ", "
        << dim2;
    if (view::IsViewApplicable(x)) {
      return view::Diagonal(x, offset, p_dim1, p_dim2);
    } else {
      auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("offset");
      attrs.SetAllAttrs(offset);
      std::vector<int32_t> input_index{p_dim1, p_dim2};
      for (int32_t i = 0; i < ndims; i++) {
        if (i != p_dim1 && i != p_dim2) { input_index.push_back(i); }
      }
      std::shared_ptr<one::Tensor> d_x = JUST(Transpose(x, input_index));
      return OpInterpUtil::Dispatch<Tensor>(*op_, {d_x}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DiagonalGradFunctor {
 public:
  DiagonalGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("diagonal_grad").Input("dy").Input("in").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x, const int32_t& offset) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("offset");
    attrs.SetAllAttrs(offset);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

// Only for ddp gradient grouping
class SliceView1dContiguousFunctor {
 public:
  SliceView1dContiguousFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, int64_t start,
                           int64_t end) const {
    if (view::IsViewApplicable(x)) { return JUST(view::Slice(x, {start}, {end}, {1})); }
    return JUST(functional::Slice(x, {start}, {end}, {1}, /*enable_view_slice=*/true));
  }
};

class TensorGetItemFunctor {
 public:
  TensorGetItemFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const TensorIndex& index) const {
    if (x->is_local() && !(LazyMode::is_enabled()) && x->requires_grad() == false
        && index.size() == 1 && index[0].IsInteger()) {
      // NOTE: speed up in special case, e.g. dataloader(refer to torch)
      // function call chain of pytorch : tensor getitem -> select -> as_strided
      // function call chain of oneflow : tensor getitem -> as_strided
      return ApplySelectIndexing(x, index);
    }

    std::vector<detail::Slice> slice_indices;
    TensorTuple tensor_indices;
    std::vector<int64_t> target_dims;
    std::vector<int64_t> expand_dims;
    JUST(PrepareSliceIndices(index, *(x->shape()), &slice_indices, &tensor_indices, &expand_dims,
                             &target_dims));

    auto expand_input = x;
    for (int i = 0; i < expand_dims.size(); ++i) {
      int64_t dim = expand_dims.at(i);
      expand_input = JUST(functional::ExpandDims(expand_input, dim + i));
    }
    int64_t ndims = expand_input->shape()->NumAxes();
    CHECK_EQ_OR_RETURN(slice_indices.size(), ndims)
        << Error::RuntimeError() << "Failed to prepare slice indices.";
    Shape target_shape(DimVector(target_dims.begin(), target_dims.end()));

    std::vector<int64_t> start(ndims), end(ndims), step(ndims);
    for (int i = 0; i < ndims; ++i) {
      const auto& slice = slice_indices.at(i);
      start[i] = slice.start();
      end[i] = slice.end();
      step[i] = slice.step();
    }
    bool is_identity = [&]() {
      if (target_shape.NumAxes() == 0) { return false; }
      for (int i = 0; i < ndims; ++i) {
        if (start[i] != 0 || end[i] != expand_input->shape()->At(i) || step[i] != 1) {
          return false;
        }
      }
      return true;
    }();
    std::shared_ptr<one::Tensor> result;
    if (is_identity) {
      result = expand_input;
    } else {
      result = JUST(Slice(expand_input, start, end, step, /*enable_view_slice=*/true));
    }

    Shape shape(DimVector(target_dims.begin(), target_dims.end()));
    if (shape != *(result->shape())) { result = JUST(Reshape(result, shape)); }
    if (!tensor_indices.empty()) {
      JUST(UnifyInputAndIndicesOnDevice(x, tensor_indices));
      result = JUST(ApplyAdvancedIndexing(result, tensor_indices));
    }
    return result;
  }
};

class TensorSetItemFunctor {
 public:
  TensorSetItemFunctor() {}
  Maybe<void> operator()(const std::shared_ptr<one::Tensor>& x, const TensorIndex& index,
                         const std::shared_ptr<one::Tensor>& value) const {
    std::vector<detail::Slice> slice_indices;
    TensorTuple tensor_indices;
    std::vector<int64_t> expand_dims;
    std::vector<int64_t> target_dims;
    JUST(PrepareSliceIndices(index, *(x->shape()), &slice_indices, &tensor_indices, &expand_dims,
                             &target_dims));
    auto expand_input = x;
    if (!expand_dims.empty()) {
      CHECK_OR_RETURN(view::IsViewApplicable(x)) << "expand dims must enable view, "
                                                    "please try to set ONEFLOW_DISABLE_VIEW=0";
      for (int i = 0; i < expand_dims.size(); ++i) {
        int64_t dim = expand_dims[i];
        expand_input = JUST(functional::ExpandDims(expand_input, dim + i));
      }
    }
    int64_t ndims = expand_input->shape()->NumAxes();
    CHECK_EQ_OR_RETURN(slice_indices.size(), ndims)
        << Error::RuntimeError() << "Failed to prepare slice indices.";

    Shape target_shape(DimVector(target_dims.begin(), target_dims.end()));
    if (target_shape.Count(0) == 0) { return Maybe<void>::Ok(); }
    const auto& value_shape = value->shape();
    bool matched = [&]() {
      for (int i = 0; i < value_shape->NumAxes() - target_shape.NumAxes(); ++i) {
        if (value_shape->At(i) != 1) { return false; }
      }
      return true;
    }();
    CHECK_OR_RETURN(matched) << Error::RuntimeError() << "The tensor size mismatch. Target sizes: "
                             << target_shape.ToString()
                             << ", value sizes: " << value_shape->ToString();
    std::shared_ptr<one::Tensor> value_tensor(value);
    // TODO: replace reshape by unsqueeze with view mechanism.
    // after here, each scalar tensor will be one with one dimension.
    for (auto& tensor : tensor_indices) {
      if (tensor && tensor->ndim() == 0) { tensor = JUST(functional::Reshape(tensor, Shape({1}))); }
    }

    DimVector slice_dims(ndims);
    std::vector<int64_t> start(ndims), end(ndims), step(ndims);
    for (int i = 0; i < ndims; ++i) {
      const auto& slice = slice_indices[i];
      start[i] = slice.start();
      end[i] = slice.end();
      step[i] = slice.step();
      slice_dims[i] = (end[i] - start[i] + step[i] - 1) / step[i];
    }
    if (tensor_indices.empty()) {
      Shape slice_shape(slice_dims);
      if (slice_shape != *(value_tensor->shape())) {
        // NOTE:
        // 1. The value shape must can be broadcasted to the target shape.
        // 2. The slice shape must have equal element count with the target shape.
        //
        // So, we should be expand to target_shape and then reshape to slice_shape.
        //
        // For example:
        // x = flow.rand(2, 3, 4)
        // y = flow.rand(3)
        // x[:, :, 1] = y
        //
        // value_shape = (3,), target_shape = (2, 3), slice_shape = (2, 3, 1)
        // We must change value shape to slice_shape if it uses SliceUpdate op.
        if (target_shape != *(value_tensor->shape()) && target_shape.NumAxes() > 0) {
          value_tensor = JUST(Expand(value_tensor, target_shape));
        }
        if (slice_shape != *(value_tensor->shape())) {
          value_tensor = JUST(Reshape(value_tensor, slice_shape));
        }
      }
      JUST(SliceUpdate(expand_input, value_tensor, start, end, step, /*inplace=*/true));
    } else {
      bool is_identity = [&]() {
        if (target_shape.NumAxes() == 0) { return false; }
        for (int i = 0; i < ndims; ++i) {
          if (start[i] != 0 || end[i] != expand_input->shape()->At(i) || step[i] != 1) {
            return false;
          }
        }
        return true;
      }();
      std::shared_ptr<one::Tensor> result;
      if (is_identity) {
        result = expand_input;
      } else {
        if (expand_input->is_local()) {
          CHECK_OR_RETURN(view::IsViewApplicable(expand_input))
              << "combined slice setitem must enable view, please try to set "
                 "ONEFLOW_DISABLE_VIEW=0";
          result = JUST(Slice(expand_input, start, end, step, /*enable_view_slice=*/true));
        } else {
          // global tensor
          result = JUST(Slice(expand_input, start, end, step, /*enable_view_slice=*/false));
        }
      }
      const Shape& slice_result_shape = *(result->shape());
      if (target_shape != slice_result_shape) {
        result = JUST(functional::View(result, target_shape));
      }

      JUST(UnifyInputAndIndicesOnDevice(result, tensor_indices));
      result = JUST(ApplyAdvancedIndexingUpdate(result, tensor_indices, value));

      // Write the sliced tensor back to the original tensor.
      if (result->is_global()) {
        if (*result->shape() != slice_result_shape) {
          CHECK_EQ_OR_RETURN(result->shape()->elem_cnt(), slice_result_shape.elem_cnt())
              << Error::RuntimeError()
              << "The global tensor size mismatch. Target sizes: " << slice_result_shape.ToString()
              << ", value sizes: " << result->shape()->ToString();
          result = JUST(functional::View(result, slice_result_shape));
        }
        JUST(SliceUpdate(expand_input, result, start, end, step, /*inplace=*/true));
      }
    }
    return Maybe<void>::Ok();
  }
};

class CastLikeFunctor {
 public:
  CastLikeFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("cast_like").Input("in").Input("dtype_like").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& like) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, like});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ElementwiseMinimumGradFunctor {
 public:
  ElementwiseMinimumGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("elementwise_minimum_backward")
                         .Input("dz")
                         .Input("x")
                         .Input("y")
                         .Output("dx")
                         .Output("dy")
                         .Build());
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& dz,
                                const std::shared_ptr<one::Tensor>& x,
                                const std::shared_ptr<one::Tensor>& y) const {
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {dz, x, y});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ElementwiseMaximumGradFunctor {
 public:
  ElementwiseMaximumGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("elementwise_maximum_backward")
                         .Input("dz")
                         .Input("x")
                         .Input("y")
                         .Output("dx")
                         .Output("dy")
                         .Build());
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& dz,
                                const std::shared_ptr<one::Tensor>& x,
                                const std::shared_ptr<one::Tensor>& y) const {
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {dz, x, y});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DivGradFunctor {
 public:
  DivGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("broadcast_div_grad")
                         .Input("dz")
                         .Input("z")
                         .Input("y")
                         .Output("dy")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dz,
                           const std::shared_ptr<one::Tensor>& z,
                           const std::shared_ptr<one::Tensor>& y) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dz, z, y});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class BroadcastPowXGradFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y,
                           const std::shared_ptr<one::Tensor>& dz) const {
    auto y_sub_one = JUST(functional::ScalarSub(y, 1, /*alpha=*/1, /*inplace=*/false));
    auto result = functional::sequence_function(functional::BroadcastPow)
                      .then(std::bind(functional::Mul, std::placeholders::_1, y))
                      .then(std::bind(functional::Mul, std::placeholders::_1, dz))
                      .then(std::bind(functional::BroadcastReduceSumLike, std::placeholders::_1, x))
                      .call(x, y_sub_one);
    return result;
  }
};

class BroadcastPowYGradFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y,
                           const std::shared_ptr<one::Tensor>& dz) const {
    auto result =
        functional::sequence_function(functional::BroadcastPow)
            .then(std::bind(functional::Mul, std::placeholders::_1, JUST(functional::Log(x))))
            .then(std::bind(functional::Mul, std::placeholders::_1, dz))
            .then(std::bind(functional::BroadcastReduceSumLike, std::placeholders::_1, y))
            .call(x, y);
    return result;
  }
};

class IdentityFunctor {
 public:
  IdentityFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("identity").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& in) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {in});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class AmpWhiteIdentityFunctor {
 public:
  AmpWhiteIdentityFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("amp_white_identity").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& in) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {in});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class AmpBlackIdentityFunctor {
 public:
  AmpBlackIdentityFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("amp_black_identity").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& in) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {in});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ReduceSumLikeFunctor {
 public:
  ReduceSumLikeFunctor() {
    op_ =
        CHECK_JUST(one::OpBuilder("reduce_sum_like").Input("x").Input("like").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& like,
                           const std::vector<int32_t>& axis) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("axis");
    attrs.SetAllAttrs(axis);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, JUST(like->detach())}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class BroadcastReduceSumLikeFunctor {
 public:
  BroadcastReduceSumLikeFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& input,
                           const std::shared_ptr<Tensor>& like) const {
    const auto& in_shape = *(input->shape());
    const auto& like_shape = *(like->shape());
    if (in_shape != like_shape) {
      const Shape& left_extended_shape =
          CreateLeftExtendedShape(ShapeView(like_shape), in_shape.NumAxes());
      if (in_shape == left_extended_shape) {
        return JUST(ReshapeLike(input, like));
      } else {
        const AxisVector& broadcast_axis_vec = left_extended_shape.Axes4BroadcastTo(in_shape);
        return JUST(ReduceSumLike(
            input, like,
            std::vector<int32_t>{broadcast_axis_vec.begin(), broadcast_axis_vec.end()}));
      }
    }
    return JUST(Identity(input));
  }
};

class SplitFunctor {
 public:
  SplitFunctor() {}
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& x,
                                const int64_t& split_size_or_sections, const int64_t& dim) const {
    int64_t axis = dim;
    axis = JUST(maybe_wrap_dim(axis, x->ndim()));
    CHECK_GE_OR_RETURN(split_size_or_sections, 0)
        << Error::RuntimeError() << "split expects split_size be non-negative, but got split_size="
        << split_size_or_sections;
    int64_t dim_size = x->shape()->At(axis);
    int64_t num_splits =
        std::max<int64_t>((dim_size + split_size_or_sections - 1) / split_size_or_sections, 1);
    TensorTuple splits(num_splits);
    int64_t last_split_size =
        split_size_or_sections - (split_size_or_sections * num_splits - dim_size);
    for (int i = 0; i < num_splits; ++i) {
      int64_t length = i < num_splits - 1 ? split_size_or_sections : last_split_size;
      splits[i] = JUST(Narrow(x, axis, i * split_size_or_sections, length));
    }
    return splits;
  }
};

class UnbindFunctor {
 public:
  UnbindFunctor() {}
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& x, const int64_t& dim) const {
    int32_t axis = dim;
    const int32_t ndim = x->ndim();
    axis = JUST(maybe_wrap_dim(axis, ndim));
    int32_t dim_size = x->shape()->At(axis);
    std::shared_ptr<TensorTuple> chunk_res = JUST(functional::Chunk(x, dim_size, axis));
    TensorTuple unbinds(dim_size);
    std::vector<int32_t> dims = {axis};
    for (int i = 0; i < dim_size; ++i) {
      unbinds[i] = JUST(functional::Squeeze((*chunk_res)[i], dims));
    }
    return unbinds;
  }
};

class ChunkFunctor {
 public:
  ChunkFunctor() {}
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& x, const int64_t& chunks,
                                const int64_t& dim) const {
    const int64_t ndim = x->ndim();
    int64_t infferd_dim = dim;
    CHECK_OR_RETURN(ndim > 0) << Error::RuntimeError()
                              << "chunk expects at least a 1-dimensional tensor.";
    CHECK_OR_RETURN(chunks > 0) << Error::RuntimeError()
                                << "chunk expects `chunks` to be greater than 0, got: " << chunks;
    infferd_dim = JUST(maybe_wrap_dim(infferd_dim, ndim));

    const auto dim_size = x->shape()->At(infferd_dim);
    int64_t split_size = (dim_size + chunks - 1) / chunks;
    if (split_size == 0 && dim_size == 0) {
      std::vector<int64_t> split_sizes(chunks, split_size);
      split_sizes[chunks - 1] = split_size - (split_size * chunks - dim_size);
      return functional::SplitWithSize(x, split_sizes, infferd_dim);
    } else {
      return functional::Split(x, split_size, infferd_dim);
    }
  }
};

class SplitLikeFunctor {
 public:
  SplitLikeFunctor() {
    ops_.resize(kMaxInputCount);
    for (int n = 1; n < ops_.size(); ++n) {
      ops_[n] = CHECK_JUST(one::OpBuilder("split_like")
                               .Input("in")
                               .Input("like", n + 1)
                               .Output("out", n + 1)
                               .Build());
    }
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& x, const TensorTuple& like,
                                const int64_t& axis) const {
    CHECK_GE_OR_RETURN(like.size(), 2)
        << Error::RuntimeError() << "like.size() must not less than 2, but got " << like.size();
    CHECK_LE_OR_RETURN(like.size(), kMaxInputCount)
        << Error::RuntimeError() << "like.size() must not greater than " << kMaxInputCount
        << ", but got " << like.size();
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("axis");
    attrs.SetAllAttrs(axis);
    TensorTuple inputs(like.size() + 1);
    inputs[0] = x;
    for (int i = 0; i < like.size(); ++i) { inputs[i + 1] = JUST(like[i]->detach()); }
    return OpInterpUtil::Dispatch<TensorTuple>(*ops_.at(like.size() - 1), inputs, attrs);
  }

 private:
  std::vector<std::shared_ptr<OpExpr>> ops_;
};

class SplitWithSizeFunctor {
 public:
  SplitWithSizeFunctor() {}
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& x,
                                const std::vector<int64_t>& split_size_or_sections,
                                const int64_t& dim) const {
    int64_t axis = dim;
    axis = JUST(maybe_wrap_dim(axis, x->ndim()));
    int64_t dim_size = x->shape()->At(axis);
    int64_t num_splits = split_size_or_sections.size();
    TensorTuple splits(num_splits);
    int64_t start_idx = 0;
    for (int i = 0; i < num_splits; ++i) {
      int64_t length = split_size_or_sections[i];
      CHECK_GE_OR_RETURN(length, 0) << Error::RuntimeError()
                                    << "split_with_sizes expects split_sizes have only "
                                       "non-negative entries, but split_sizes["
                                    << i << "] = " << length;
      splits[i] = JUST(Narrow(x, axis, start_idx, length));
      start_idx += length;
    }
    CHECK_EQ_OR_RETURN(start_idx, dim_size)
        << Error::RuntimeError() << "split_with_sizes expects split_sizes to sum exactly to "
        << dim_size << " (input tensor's size at dimension " << axis << "), "
        << "but got sum(split_sizes)=" << start_idx;
    return splits;
  }
};

class BatchGatherFunctor {
 public:
  BatchGatherFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("batch_gather").Input("in").Input("indices").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& in,
                           const std::shared_ptr<one::Tensor>& indices) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {in, indices});
  }

 protected:
  std::shared_ptr<OpExpr> op_;
};

class UnsortedBatchSegmentSumFunctor {
 public:
  UnsortedBatchSegmentSumFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("unsorted_batch_segment_sum")
                         .Input("data")
                         .Input("segment_ids")
                         .Output("out")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& data,
                           const std::shared_ptr<one::Tensor>& segment_ids,
                           const int64_t& num_segments) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("num_segments");
    attrs.SetAllAttrs(num_segments);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {data, segment_ids}, attrs);
  }

 protected:
  std::shared_ptr<OpExpr> op_;
};

template<bool inplace>
class MaskedFillFunctor {
 public:
  MaskedFillFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("masked_fill").Input("x").Input("mask").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& mask, const Scalar& value) const {
    auto& attrs =
        THREAD_CACHED_MUTABLE_ATTR_MAP("float_operand", "has_float_operand", "int_operand",
                                       "has_int_operand", "bool_operand", "has_bool_operand");
    if (IsFloatingDataType(x->dtype()->data_type())) {
      attrs.SetAllAttrs(value.As<double>(), true, NullOpt, false, NullOpt, false);
    } else if (IsIntegralDataType(x->dtype()->data_type())) {
      attrs.SetAllAttrs(NullOpt, false, value.As<int64_t>(), true, NullOpt, false);
    } else if (IsBoolDataType(x->dtype()->data_type())) {
      attrs.SetAllAttrs(NullOpt, false, NullOpt, false, value.As<bool>(), true);
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "Only support floating or integral data type.";
    }
    const auto& x_shape = *(x->shape());
    const auto& mask_shape = *(mask->shape());

    std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
    if (inplace) {
      JUST(CheckInplaceValid(x));
      (*outputs)[0] = x;
    }

    if (x_shape != mask_shape) {
      Shape max_shape = Shape::Ones(std::max(x_shape.NumAxes(), mask_shape.NumAxes()));
      const Shape& x_extend_shape =
          CreateLeftExtendedShape(ShapeView(x_shape), max_shape.NumAxes());
      const Shape& mask_extend_shape =
          CreateLeftExtendedShape(ShapeView(mask_shape), max_shape.NumAxes());
      FOR_RANGE(int64_t, i, 0, max_shape.NumAxes()) {
        max_shape.Set(i, std::max(x_extend_shape.At(i), mask_extend_shape.At(i)));
      }
      JUST(OpInterpUtil::Dispatch(*op_, {JUST(Expand(x, max_shape)), JUST(Expand(mask, max_shape))},
                                  outputs.get(), attrs));
      return outputs->at(0);
    }

    JUST(OpInterpUtil::Dispatch(*op_, {x, mask}, outputs.get(), attrs));
    return outputs->at(0);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class MeshgridFunctor {
 public:
  Maybe<TensorTuple> operator()(const TensorTuple& tensors, const std::string& indexing) const {
    int size = tensors.size();
    CHECK_GT_OR_RETURN(size, 0) << Error::RuntimeError()
                                << "meshgrid expects a non-empty TensorList";
    for (int i = 0; i < size - 1; ++i) {
      const auto& cur_tensor = JUST(VectorAt(tensors, i));
      const auto& next_tensor = JUST(VectorAt(tensors, i + 1));
      CHECK_OR_RETURN(cur_tensor->dtype() == next_tensor->dtype())
          << Error::RuntimeError() << "meshgrid expects all tensors to have the same dtype";
      if (cur_tensor->is_local()) {
        CHECK_OR_RETURN(next_tensor->is_local())
            << Error::RuntimeError() << "meshgrid expects all tensors are local tensor";
        CHECK_OR_RETURN(JUST(cur_tensor->device())->type() == JUST(next_tensor->device())->type())
            << Error::RuntimeError() << "meshgrid expects all tensors to have the same device";
      } else {
        CHECK_OR_RETURN(!next_tensor->is_local())
            << Error::RuntimeError() << "meshgrid expects all tensors are global tensor";
        CHECK_OR_RETURN(JUST(cur_tensor->parallel_desc()) == JUST(next_tensor->parallel_desc()))
            << Error::RuntimeError() << "meshgrid expects all tensors to have the same placement";
      }
    }

    std::vector<std::shared_ptr<Tensor>> tensor_consts(tensors.begin(), tensors.end());

    bool swap_first_and_second_tensors = false;
    if (indexing == "xy") {
      swap_first_and_second_tensors = (size >= 2);
      if (swap_first_and_second_tensors) { std::swap(tensor_consts[0], tensor_consts[1]); }
    } else {
      CHECK_EQ_OR_RETURN(indexing, "ij") << Error::RuntimeError()
                                         << "meshgrid: indexing must be one of \"xy\" or \"ij\", "
                                            "but received: "
                                         << indexing;
    }

    TensorTuple grids(size);
    DimVector grids_vec(size);
    for (int i = 0; i < size; ++i) {
      CHECK_LE_OR_RETURN(tensor_consts[i]->shape()->NumAxes(), 1)
          << Error::RuntimeError() << "Expected scalar or 1D tensor in the tensor list but got "
          << tensor_consts[i]->shape()->NumAxes();
      if (tensor_consts[i]->shape()->NumAxes() == 0) {
        grids_vec[i] = 1;
      } else {
        grids_vec[i] = tensor_consts[i]->shape()->At(0);
      }
    }
    Shape grids_shape(grids_vec);

    DimVector view_shape_vec(size, 1);
    Shape view_shape(view_shape_vec);
    for (int i = 0; i < size; ++i) {
      view_shape.Set(i, -1);
      std::shared_ptr<one::Tensor> reshaped = JUST(Reshape(tensor_consts.at(i), view_shape));
      grids[i] = JUST(Expand(reshaped, grids_shape));
      view_shape.Set(i, 1);
    }

    if (swap_first_and_second_tensors) { std::swap(grids[0], grids[1]); }

    return grids;
  }
};

class IndexSelectFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int64_t& dim,
                           const std::shared_ptr<one::Tensor>& index) const {
    const int64_t input_num_axes = input->shape()->NumAxes();
    const int64_t index_num_axes = index->shape()->NumAxes();
    CHECK_LE_OR_RETURN(index_num_axes, 1)
        << Error::IndexError() << "index_select(): Index is supposed to be a vector";
    bool index_dtype_flag =
        (index->dtype()->data_type() == kInt32) || (index->dtype()->data_type() == kInt64);
    CHECK_EQ_OR_RETURN(index_dtype_flag, true)
        << Error::RuntimeError() << "index_select(): Expected dtype int32 or int64 for index";
    int64_t new_dim = dim;
    new_dim = JUST(maybe_wrap_dim(new_dim, input_num_axes));
    return JUST(functional::Gather(input, index, new_dim));
  }
};

namespace {

Maybe<Tensor> LocalTensorTo(const std::shared_ptr<Tensor>& x, Symbol<Device> device,
                            const Symbol<DType>& dtype, const bool& copy) {
  std::shared_ptr<Tensor> tensor = x;
  if (device != JUST(x->device())) { tensor = JUST(Copy(tensor, device, /*pin_memory=*/false)); }
  if (dtype != x->dtype()) { tensor = JUST(Cast(tensor, dtype, /*pin_memory=*/false)); }
  if (copy && tensor == x) { tensor = JUST(Copy(tensor, device, /*pin_memory=*/false)); }
  return tensor;
}

Maybe<Tensor> GlobalTensorTo(const std::shared_ptr<Tensor>& x, const std::string& device_type,
                             const Symbol<DType>& dtype, const bool& copy) {
  std::shared_ptr<Tensor> tensor;
  auto input_placement = JUST(x->parallel_desc());
  std::string input_device_tag = input_placement->device_tag();
  if (input_device_tag == "gpu") { input_device_tag = "cuda"; }
  if (device_type == input_device_tag) {
    if (dtype == x->dtype()) {
      return (copy ? JUST(x->clone()) : x);
    } else {
      return JUST(Cast(x, dtype, /*pin_memory=*/false));
    }
  }
  if (LazyMode::is_enabled()) {
    if (dtype != x->dtype()) { tensor = JUST(Cast(x, dtype, /*pin_memory=*/false)); }
    if (device_type != JUST(x->parallel_desc())->device_tag()) {
      tensor = JUST(Copy(tensor ? tensor : x, device_type, 0, /*pin_memory=*/false));
    }
    return tensor;
  } else {
    CheckMetaConsistency(x).GetOrThrow();
    auto placement = JUST(ReplacePlacementDeviceTag(input_placement, device_type));
    auto nd_sbp = JUST(x->nd_sbp());
    std::vector<Symbol<SbpParallel>> sbp_tuple(nd_sbp->sbp_parallel().size());
    for (int i = 0; i < sbp_tuple.size(); ++i) { sbp_tuple[i] = nd_sbp->sbp_parallel().Get(i); }
    tensor = JUST(GlobalToLocal(x, /*copy=*/false));
    Symbol<Device> device = JUST(Device::New(device_type));
    tensor = JUST(LocalTensorTo(tensor, device, dtype, copy));
    JUST(tensor->set_requires_grad(x->requires_grad()));
    return JUST(LocalToGlobal(tensor, placement, sbp_tuple, *(x->shape()), dtype,
                              /* sync_data */ true, /*copy=*/false));
  }
}

}  // namespace

class ToFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& input,
                           const Optional<std::string>& device_,
                           const Optional<Symbol<DType>>& dtype_, bool copy) const {
    Symbol<DType> dtype = dtype_.value_or(input->dtype());
    if (input->is_global()) {
      std::string device_type = device_.value_or(JUST(input->parallel_desc())->device_tag());
      CHECK_OR_RETURN(ep::DeviceManagerRegistry::GetDeviceTypeByDeviceTypeName(device_type)
                      != DeviceType::kInvalidDevice)
          << Error::RuntimeError()
          << "Only string device without device id (eg. \"cpu\" or \"cuda\") is expected "
          << "for global tensor, but got " << device_.value_or("");
      return JUST(GlobalTensorTo(input, device_type, dtype, copy));
    } else {
      Symbol<Device> device =
          device_
              .map([](const std::shared_ptr<std::string>& str) -> Symbol<Device> {
                return CHECK_JUST(Device::ParseAndNew(*str));
              })
              .value_or(JUST(input->device()));
      return JUST(LocalTensorTo(input, device, dtype, copy));
    }
  }
};

class To2Functor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& input,
                           const Optional<Symbol<Device>>& device_,
                           const Optional<Symbol<DType>>& dtype_, bool copy) const {
    if (input->is_global()) {
      if (!device_.has_value()) {
        std::string device_type = JUST(input->parallel_desc())->device_tag();
        return JUST(GlobalTensorTo(input, device_type, dtype_.value_or(input->dtype()), copy));
      } else {
        if (!GlobalMode::is_enabled()) {
          CHECK_OR_RETURN(!device_.has_value())
              << Error::RuntimeError()
              << "Only string device without device id (eg. \"cpu\" or \"cuda\") is expected "
              << "for global tensor, but got " << device_.value_or(Symbol<Device>())->ToRepr();
        }
        std::string device_type = device_.value_or(Symbol<Device>())->type();
        return JUST(GlobalTensorTo(input, device_type, dtype_.value_or(input->dtype()), copy));
      }
    } else {
      auto dtype = dtype_.value_or(input->dtype());
      auto device = device_.value_or(JUST(input->device()));
      return JUST(LocalTensorTo(input, device, dtype, copy));
    }
  }
};

class To3Functor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& input,
                           const Optional<Symbol<DType>>& dtype_, bool copy) const {
    Symbol<DType> dtype = dtype_.value_or(input->dtype());
    if (input->is_global()) {
      return GlobalTensorTo(input, JUST(input->parallel_desc())->device_tag(), dtype, copy);
    } else {
      auto device = JUST(input->device());
      return LocalTensorTo(input, device, dtype, copy);
    }
  }
};

class To4Functor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& input,
                           const std::shared_ptr<Tensor>& other, bool copy) const {
    CHECK_OR_RETURN(!input->is_global() && !other->is_global())
        << Error::RuntimeError()
        << "tensor.to(other) can only be called when tensor and other are local tensors";
    Symbol<DType> dtype = other->dtype();
    Symbol<Device> device = JUST(other->device());
    return LocalTensorTo(input, device, dtype, copy);
  }
};

class ToDeviceFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& input,
                           const Optional<std::string>& device_) const {
    Symbol<DType> dtype = input->dtype();
    const bool copy = false;
    if (input->is_global()) {
      std::string device_type = device_.value_or(JUST(input->parallel_desc())->device_tag());
      CHECK_OR_RETURN(ep::DeviceManagerRegistry::GetDeviceTypeByDeviceTypeName(device_type)
                      != DeviceType::kInvalidDevice)
          << Error::RuntimeError()
          << "Only string device without device id (eg. \"cpu\" or \"cuda\") is expected "
          << "for global tensor, but got " << device_.value_or("");
      return JUST(GlobalTensorTo(input, device_type, dtype, copy));
    } else {
      Symbol<Device> device =
          device_
              .map([](const std::shared_ptr<std::string>& str) -> Symbol<Device> {
                return CHECK_JUST(Device::ParseAndNew(*str));
              })
              .value_or(JUST(input->device()));
      return JUST(LocalTensorTo(input, device, dtype, copy));
    }
  }
};

class TopKFunctor {
 public:
  TopKFunctor() { op_ = CHECK_JUST(one::OpBuilder("top_k").Input("in").Output("out").Build()); }
  Maybe<TensorTuple> operator()(const std::shared_ptr<Tensor>& input, const int32_t k,
                                const Optional<int32_t>& dim, const bool largest,
                                const bool sorted) const {
    auto outputs = std::make_shared<TensorTuple>(2);
    std::shared_ptr<Tensor> values;
    std::shared_ptr<Tensor> indices;

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("k", "sorted");
    attrs.SetAllAttrs(k, sorted);

    int32_t dim_value = dim.value_or(-1);
    int32_t axis = dim_value;
    axis = JUST(maybe_wrap_dim(axis, input->ndim()));
    if (axis == input->ndim() - 1) {
      if (largest) {
        indices = JUST(OpInterpUtil::Dispatch<Tensor>(*op_, {input}, attrs));
      } else {
        auto neg_input = JUST(ScalarMul(input, -1, false));
        indices = JUST(OpInterpUtil::Dispatch<Tensor>(*op_, {neg_input}, attrs));
      }
      values = JUST(DimGather(input, axis, indices, false));

    } else {
      auto perm = JUST(GetPermWhenTransposeAxisToLastDim(input->ndim(), dim_value));
      auto x = JUST(Transpose(input, *perm));
      if (largest) {
        indices = JUST(OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs));
      } else {
        auto neg_input = JUST(ScalarMul(x, -1, false));
        indices = JUST(OpInterpUtil::Dispatch<Tensor>(*op_, {neg_input}, attrs));
      }
      auto inversed_perm = JUST(GetInversedPerm(*perm));
      indices = JUST(Transpose(indices, *inversed_perm));
      values = JUST(DimGather(input, axis, indices, false));
    }
    (*outputs)[0] = values;
    (*outputs)[1] = indices;
    return outputs;
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class InTopKFunctor {
 public:
  InTopKFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("in_top_k").Input("targets").Input("predictions").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& targets,
                           const std::shared_ptr<Tensor>& predictions, int32_t k) const {
    CHECK_EQ_OR_RETURN(targets->shape()->At(0), predictions->shape()->At(0))
        << Error::RuntimeError() << "The num of targets must equal the num of predictions";
    CHECK_EQ_OR_RETURN(targets->ndim(), 1)
        << Error::RuntimeError() << "The dimension of targets must be 1";
    CHECK_EQ_OR_RETURN(predictions->ndim(), 2)
        << Error::RuntimeError() << "The dimension of predictions must be 2";
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("k");
    attrs.SetAllAttrs(k);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {targets, predictions}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class TensorBufferToTensorFunctor {
 public:
  TensorBufferToTensorFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("tensor_buffer_to_tensor").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& input, const Shape& instance_shape,
                           const Symbol<DType>& dtype) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("instance_shape", "dtype");
    attrs.SetAllAttrs(instance_shape, dtype->data_type());
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class TensorToTensorBufferFunctor {
 public:
  TensorToTensorBufferFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("tensor_to_tensor_buffer").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& input, int32_t instance_dims) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("instance_dims");
    attrs.SetAllAttrs(instance_dims);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class GenTensorBufferFunctor {
 public:
  GenTensorBufferFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("gen_tensor_buffer").Output("out").Build());
  }
  Maybe<Tensor> operator()(const Shape& shape, const std::vector<Shape>& shape_list,
                           const std::vector<float>& value_list, const Symbol<DType>& dtype,
                           bool dynamic_out) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("shape", "shape_list", "value_list", "data_type",
                                                 "dynamic_out");
    attrs.SetAllAttrs(shape, shape_list, value_list, dtype->data_type(), dynamic_out);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class RepeatFunctor {
 public:
  RepeatFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const Shape& repeat_shape) const {
    Shape input_shape = *(input->shape());
    std::vector<int32_t> input_reshape_vec;
    std::vector<int32_t> expand_shape_vec;
    std::vector<int32_t> output_reshape_vec;

    int32_t numaxes_diff = repeat_shape.NumAxes() - input_shape.NumAxes();
    CHECK_GE_OR_RETURN(numaxes_diff, 0) << Error::RuntimeError()
                                        << "Number of dimensions of repeat dims can not be "
                                           "smaller than number of dimensions of tensor";

    for (int32_t i = repeat_shape.NumAxes() - 1; i >= 0; i--) {
      if (i >= numaxes_diff) {
        int32_t input_shape_val = input_shape.At(i - numaxes_diff);
        int32_t repeat_shape_val = repeat_shape.At(i);
        if (repeat_shape_val > 1) {
          if (input_shape_val > 1) {
            input_reshape_vec.insert(input_reshape_vec.begin(), input_shape_val);
            input_reshape_vec.insert(input_reshape_vec.begin(), 1);
            expand_shape_vec.insert(expand_shape_vec.begin(), input_shape_val);
            expand_shape_vec.insert(expand_shape_vec.begin(), repeat_shape_val);
            output_reshape_vec.insert(output_reshape_vec.begin(),
                                      repeat_shape_val * input_shape_val);
          } else {
            input_reshape_vec.insert(input_reshape_vec.begin(), input_shape_val);
            expand_shape_vec.insert(expand_shape_vec.begin(), repeat_shape_val);
            output_reshape_vec.insert(output_reshape_vec.begin(), repeat_shape_val);
          }
        } else {
          input_reshape_vec.insert(input_reshape_vec.begin(), input_shape_val);
          // For 0-size tensor, align with PyTorch.
          if (repeat_shape_val == 0) {
            expand_shape_vec.insert(expand_shape_vec.begin(), 0);
            output_reshape_vec.insert(output_reshape_vec.begin(), 0);
          } else {
            expand_shape_vec.insert(expand_shape_vec.begin(), input_shape_val);
            output_reshape_vec.insert(output_reshape_vec.begin(), input_shape_val);
          }
        }
      } else {
        expand_shape_vec.insert(expand_shape_vec.begin(), repeat_shape.At(i));
        output_reshape_vec.insert(output_reshape_vec.begin(), repeat_shape.At(i));
      }
    }
    Shape input_reshape(DimVector(input_reshape_vec.begin(), input_reshape_vec.end()));
    Shape expand_shape(DimVector(expand_shape_vec.begin(), expand_shape_vec.end()));
    Shape output_reshape(DimVector(output_reshape_vec.begin(), output_reshape_vec.end()));
    std::shared_ptr<one::Tensor> reshaped_tensor = JUST(Reshape(input, input_reshape));
    std::shared_ptr<one::Tensor> expanded_tensor = JUST(Expand(reshaped_tensor, expand_shape));
    std::shared_ptr<one::Tensor> result = JUST(Reshape(expanded_tensor, output_reshape));
    return result->contiguous();
  }
};

class RepeatInterLeaveIndexFunctor {
 public:
  RepeatInterLeaveIndexFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("repeat_interleave").Input("in").Input("cumsum").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& cumsum,
                           const int32_t& repeat_num) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("repeat_num");
    attrs.SetAllAttrs(static_cast<int64_t>(repeat_num));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input, cumsum}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class RepeatInterLeaveIntFunctor {
 public:
  RepeatInterLeaveIntFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int32_t& repeats,
                           const Optional<int32_t>& dim) const {
    CHECK_OR_RETURN(input->is_local() == true)
        << Error::RuntimeError() << "repeat_interleave only support local tensor now";
    std::shared_ptr<one::Tensor> res;
    if (!dim.has_value()) {
      std::shared_ptr<one::Tensor> flatten_input = JUST(Flatten(input, 0, -1));
      std::shared_ptr<one::Tensor> repeats_expand = JUST(
          Expand(JUST(Constant(Shape{1}, Scalar(repeats), DType::Int32(), JUST(input->device()))),
                 Shape{flatten_input->shape()->At(0)}));
      std::shared_ptr<one::Tensor> cumsum = JUST(Cumsum(repeats_expand, 0, DType::Int32()));
      int64_t output_size = flatten_input->shape()->At(0);
      if (repeats > 0) { output_size *= repeats; }
      res = JUST(IndexSelect(flatten_input, 0,
                             JUST(RepeatInterLeaveIndex(repeats_expand, cumsum, output_size))));
    } else {
      int32_t dim_ = JUST(dim);
      const auto& input_shape = input->shape();
      const int64_t& num_axes = input_shape->NumAxes();
      dim_ = JUST(maybe_wrap_dim(dim_, num_axes));
      std::shared_ptr<one::Tensor> repeats_expand = JUST(
          Expand(JUST(Constant(Shape{1}, Scalar(repeats), DType::Int32(), JUST(input->device()))),
                 Shape{input->shape()->At(dim_)}));
      std::shared_ptr<one::Tensor> cumsum = JUST(Cumsum(repeats_expand, 0, DType::Int32()));
      int64_t output_size = input->shape()->At(dim_);
      if (repeats > 0) { output_size *= repeats; }
      res = JUST(IndexSelect(input, dim_,
                             JUST(RepeatInterLeaveIndex(repeats_expand, cumsum, output_size))));
    }
    return res;
  }
};

class RepeatInterLeaveTensorFunctor {
 public:
  RepeatInterLeaveTensorFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& repeats, const int32_t& dim,
                           const Optional<int32_t>& output_size) const {
    CHECK_OR_RETURN(input->is_local() == true)
        << Error::RuntimeError() << "repeat_interleave only support local tensor now";
    const auto repeats_shape = repeats->shape();
    const int64_t& repeat_num_axes = repeats_shape->NumAxes();
    CHECK_OR_RETURN(repeat_num_axes == 1)
        << Error::RuntimeError() << "repeat_interleave only accept 1D vector as repeat";
    CHECK_OR_RETURN(repeats->dtype() == DType::Int64())
        << Error::RuntimeError() << "repeats has to be Long tensor";

    std::vector<int64_t> repeats_value(repeats_shape->elem_cnt());
    if (!output_size.has_value()) {
      const auto& callback = [&](ep::Stream* stream,
                                 const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
        SyncAutoMemcpy(stream, repeats_value.data(), eager_blob_object->dptr(),
                       repeats_value.size() * sizeof(int64_t), memory::MakeHostMemCase(),
                       eager_blob_object->mem_case());
      };
      SyncAccessTensorWithTimeOut(repeats, callback, "const").GetOrThrow();
      for (const auto x : repeats_value) {
        CHECK_OR_RETURN(x >= 0) << Error::RuntimeError() << "repeats can not be negative";
      }
    } else {
      repeats_value.push_back(JUST(output_size));
    }
    int32_t dim_ = dim;
    const auto& input_shape = input->shape();
    const int64_t& num_axes = input_shape->NumAxes();
    dim_ = JUST(maybe_wrap_dim(dim_, num_axes));
    CHECK_OR_RETURN(repeats_shape->At(0) == input->shape()->At(dim_))
        << Error::RuntimeError() << "repeats must have the same size as input along dim";
    std::shared_ptr<one::Tensor> cumsum = JUST(Cumsum(repeats, 0, DType::Int32()));
    const int64_t& output_size_value =
        std::accumulate(repeats_value.begin(), repeats_value.end(), 0);
    return JUST(
        IndexSelect(input, dim_, JUST(RepeatInterLeaveIndex(repeats, cumsum, output_size_value))));
  }
};

class TileFunctor {
 public:
  TileFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const Shape& dims) const {
    std::vector<int32_t> new_dims_vec;
    int32_t numaxes_diff = input->shape()->NumAxes() - dims.NumAxes();
    for (int32_t i = dims.NumAxes() - 1; i >= 0; i--) {
      CHECK_GE_OR_RETURN(dims.At(i), 0)
          << Error::RuntimeError() << "Trying to create tensor with negative dimension "
          << dims.At(i);
      new_dims_vec.insert(new_dims_vec.begin(), dims.At(i));
    }
    for (int32_t i = 0; i < numaxes_diff; i++) { new_dims_vec.insert(new_dims_vec.begin(), 1); }
    Shape new_dims(DimVector(new_dims_vec.begin(), new_dims_vec.end()));
    return JUST(Repeat(input, new_dims));
  }
};

class TransposeAllDimPropertyFunctor {
 public:
  TransposeAllDimPropertyFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x) const {
    const int64_t ndim = x->ndim();
    std::vector<int32_t> permute;
    permute.resize(ndim);
    std::iota(permute.begin(), permute.end(), 0);
    std::reverse(permute.begin(), permute.end());
    return Transpose(x, permute);
  }
};

class TransposeAllDimFunctionFunctor {
 public:
  TransposeAllDimFunctionFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x) const {
    const int64_t ndim = x->ndim();
    CHECK_OR_RETURN(ndim <= 2) << Error::RuntimeError()
                               << "t() expects a tensor with <= 2 dimensions, but input tensor is "
                               << ndim << "D";
    if (ndim == 0 || ndim == 1) { return x; }
    return Transpose2dim(x, 0, 1);
  }
};

class ReshapeLikeFunctor {
 public:
  ReshapeLikeFunctor() {
    op_ =
        CHECK_JUST(one::OpBuilder("reshape_like").Input("in").Input("like").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& like) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, JUST(like->detach())});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class PinMemoryFunctor {
 public:
  PinMemoryFunctor() {
    op_ =
        CHECK_JUST(one::OpBuilder("slice_update").Input("ref").Input("value").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input) const {
    // TODO:(zhaoluyang) support global tensor.pin_memory()
    CHECK_OR_RETURN(input->is_local() && !(LazyMode::is_enabled()))
        << Error::RuntimeError() << "Tensor.pin_memory() only support local tensor for now!";
    // if tensor already pinned, then just return
    if (JUST(JUST(input->AsLocalTensor())->is_pinned())) { return input; }
    auto shape = input->shape();
    auto device = JUST(input->device());
    const bool requires_grad = input->requires_grad();
    CHECK_EQ_OR_RETURN(device->enum_type(), DeviceType::kCPU)
        << Error::RuntimeError() << "cannot pin tensor with device: " << device->ToString()
        << ", only dense CPU tensors can be pinned.";

    auto empty = JUST(functional::Empty(*shape.get(), input->dtype(), device, requires_grad,
                                        /*pin_memory=*/true));
    const int32_t ndim = input->ndim();
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("start", "stop", "step");
    if (ndim == 0) {
      // TODO(wyg): use TensorSetItem after supporting non-requires_grad tensor inplace
      // for 0-dim tensor
      empty = JUST(functional::ExpandDims(empty, 0));              // expand to [1, ]
      auto expand_input = JUST(functional::ExpandDims(input, 0));  // expand to [1, ]
      attrs.SetAllAttrs(std::vector<int64_t>{0}, std::vector<int64_t>{1}, std::vector<int64_t>{1});
      auto outputs = TensorTuple{empty};
      JUST(OpInterpUtil::Dispatch(*op_, TensorTuple{empty, expand_input}, &outputs, attrs));
      return outputs[0];
    } else {
      std::vector<int64_t> starts(ndim, 0);
      std::vector<int64_t> stops(ndim);
      std::vector<int64_t> steps(ndim, 1);
      for (int i = 0; i < ndim; ++i) { stops[i] = input->shape()->At(i); }
      attrs.SetAllAttrs(starts, stops, steps);
      JUST(empty->set_requires_grad(requires_grad));
      auto outputs = TensorTuple{empty};
      JUST(OpInterpUtil::Dispatch(*op_, TensorTuple{empty, input}, &outputs, attrs));
      return outputs[0];
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class FillFunctor {
 public:
  FillFunctor() { op_ = CHECK_JUST(one::OpBuilder("fill_").Input("in").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& in, const Scalar& value) const {
    JUST(CheckInplaceValid(in));
    auto& attrs =
        THREAD_CACHED_MUTABLE_ATTR_MAP("floating_value", "is_floating_value", "integral_value");
    if (IsFloatingDataType(in->dtype()->data_type())) {
      attrs.SetAllAttrs(value.As<double>(), true, NullOpt);
    } else if (IsIntegralDataType(in->dtype()->data_type())) {
      attrs.SetAllAttrs(NullOpt, false, value.As<int64_t>());
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "Only support floating or integral data type.";
    }
    auto outputs = std::make_shared<TensorTuple>(1);
    (*outputs)[0] = in;
    JUST(OpInterpUtil::Dispatch(*op_, {in}, outputs.get(), attrs));
    return (*outputs)[0];
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class FillTensorFunctor {
 public:
  FillTensorFunctor() {
    op_ =
        CHECK_JUST(one::OpBuilder("fill_tensor_").Input("in").Input("value").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& in,
                           const std::shared_ptr<one::Tensor>& value) const {
    JUST(CheckInplaceValid(in));
    const int64_t ndim = value->ndim();
    CHECK_EQ_OR_RETURN(ndim, 0)
        << Error::RuntimeError()
        << "fill_ only supports 0-dimension value tensor but got tensor with " << ndim
        << " dimensions.";
    TensorProcessor tensor_processor;
    JUST(tensor_processor.PromoteInputsToCommonDtype(true, in->dtype())
             .AddInputs({in, value})
             .Apply());
    TensorTuple input_tuple = JUST(tensor_processor.GetInputs());
    auto outputs = std::make_shared<TensorTuple>(1);
    (*outputs)[0] = in;
    JUST(OpInterpUtil::Dispatch(*op_, {input_tuple[0], input_tuple[1]}, outputs.get()));
    return (*outputs)[0];
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class IndexAddFunctor {
 public:
  IndexAddFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("index_add")
                         .Input("input")
                         .Input("index")
                         .Input("source")
                         .Output("output")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int64_t& dim,
                           const std::shared_ptr<one::Tensor>& index,
                           const std::shared_ptr<one::Tensor>& source, const Scalar& alpha) const {
    CHECK_OR_RETURN(source->ndim() == 0 || index->shape()->Count(0) == source->shape()->At(dim))
        << "index_copy_(): Number of indices (," << index->shape()->Count(0)
        << ", \") should be equal to source.size(dim) (," << source->shape()->At(dim) << ", \")";
    CHECK_OR_RETURN(index->dtype()->data_type() != DataType::kInt32
                    || index->dtype()->data_type() != DataType::kInt64)
        << "Input(Index) holds the wrong type, it holds "
        << DataType_Name(index->dtype()->data_type())
        << " , but "
           "desires to be int32_t or int64_t";
    const float alpha_value = alpha.As<float>();
    int64_t dim_ = dim;
    dim_ = JUST(maybe_wrap_dim(dim_, input->ndim()));
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("dim", "alpha");
    attrs.SetAllAttrs(dim_, alpha_value);
    TensorProcessor tensor_processor;
    JUST(tensor_processor.PromoteInputsToCommonDtype(true, input->dtype())
             .AddInputs({input, source})
             .Apply());
    TensorTuple input_tuple = JUST(tensor_processor.GetInputs());
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input, index, input_tuple.at(1)}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class IndexAddInplaceFunctor {
 public:
  IndexAddInplaceFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("index_add")
                         .Input("input")
                         .Input("index")
                         .Input("source")
                         .Output("output")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int64_t& dim,
                           const std::shared_ptr<one::Tensor>& index,
                           const std::shared_ptr<one::Tensor>& source, const Scalar& alpha) const {
    CHECK_OR_RETURN(source->ndim() == 0 || index->shape()->Count(0) == source->shape()->At(dim))
        << "index_copy_(): Number of indices (," << index->shape()->Count(0)
        << ", \") should be equal to source.size(dim) (," << source->shape()->At(dim) << ", \")";
    CHECK_OR_RETURN(index->dtype()->data_type() != DataType::kInt32
                    || index->dtype()->data_type() != DataType::kInt64)
        << "Input(Index) holds the wrong type, it holds "
        << DataType_Name(index->dtype()->data_type())
        << " , but "
           "desires to be int32_t or int64_t";
    const float alpha_value = alpha.As<float>();
    int64_t dim_ = dim;
    dim_ = JUST(maybe_wrap_dim(dim_, input->ndim()));
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("dim", "alpha");
    attrs.SetAllAttrs(dim_, alpha_value);
    JUST(CheckInplaceValid(input));
    std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
    outputs->at(0) = input;
    TensorProcessor tensor_processor;
    JUST(tensor_processor.PromoteInputsToCommonDtype(true, input->dtype())
             .AddInputs({input, source})
             .Apply());
    TensorTuple input_tuple = JUST(tensor_processor.GetInputs());
    JUST(OpInterpUtil::Dispatch(*op_, {input, index, input_tuple.at(1)}, outputs.get(), attrs));
    return outputs->at(0);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class BroadcastShapesFunctor {
 public:
  Maybe<Shape> operator()(const std::vector<Shape>& shapes) const {
    return InferUnifiedShapeForBroadcasting(shapes);
  }
};

class BroadcastTensorsFunctor {
 public:
  Maybe<TensorTuple> operator()(const TensorTuple& tensors) const {
    if (tensors.empty()) { return Error::RuntimeError() << "tensors should not be empty."; }

    Shape shape_to_broadcast;
    std::deque<bool> need_to_broadcast;

    std::tie(shape_to_broadcast, need_to_broadcast) =
        *JUST(InferUnifiedShapeForBroadcastingWithInfo([&tensors]() {
          std::vector<Shape> shapes;
          for (auto& x : tensors) { shapes.push_back(*x->shape()); }
          return shapes;
        }()));

    std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>();
    for (size_t i = 0; i < tensors.size(); ++i) {
      outputs->emplace_back(need_to_broadcast.at(i)  // NOLINT
                                ? JUST(functional::Expand(tensors.at(i), shape_to_broadcast))
                                : tensors.at(i));
    }
    return outputs;
  }
};
class BinCountFunctor {
 public:
  BinCountFunctor() {
    op_ = CHECK_JUST(OpBuilder("bincount").Input("in").Output("out").Build());
    weight_op_ =
        CHECK_JUST(OpBuilder("bincount").Input("in").Input("weight").Output("out").Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& input, const Optional<Tensor>& weight,
                           const Optional<int64_t>& minlength) const {
    CHECK_OR_RETURN(!input->dtype()->is_floating_point()) << "bincount can only support int tensor";
    TensorProcessor tensor_processor;
    JUST(tensor_processor.AddInputs({input}, DType::Int64()).Apply());
    const auto x = JUST(tensor_processor.GetInputs()).at(0);
    std::shared_ptr<Tensor> local_tensor = x;
    int64_t max = 0;

    // check min value
    {
      if (x->is_global()) { local_tensor = JUST(GlobalToLocal(x, false)); }
      auto tensor_min = JUST(functional::Min(local_tensor));
      int64_t min = 0;
      const auto& callback_min =
          [&](ep::Stream* stream, const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
            SyncAutoMemcpy(stream, &min, eager_blob_object->dptr(), sizeof(min),
                           memory::MakeHostMemCase(), eager_blob_object->mem_case());
          };
      JUST(SyncAccessTensorWithTimeOut(tensor_min, callback_min, "const"));
      CHECK_GE_OR_RETURN(min, 0) << "bincount only supports 1-d non-negative integral inputs.";

      auto tensor_max = JUST(functional::Max(local_tensor));
      const auto& callback_max =
          [&](ep::Stream* stream, const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
            SyncAutoMemcpy(stream, &max, eager_blob_object->dptr(), sizeof(max),
                           memory::MakeHostMemCase(), eager_blob_object->mem_case());
          };
      JUST(SyncAccessTensorWithTimeOut(tensor_max, callback_max, "const"));
      max += 1;
    }
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("size");
    if (minlength) {
      CHECK_GE_OR_RETURN(JUST(minlength), 0) << "minlength should be >= 0";
      max = std::max(JUST(minlength), max);
    }
    attrs.SetAllAttrs(max);
    if (weight) {
      CHECK_EQ_OR_RETURN(JUST(weight)->nelement(), x->nelement())
          << "input and weights should have the same length";
      return OpInterpUtil::Dispatch<Tensor>(*weight_op_, {x, JUST(weight)}, attrs);
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
  std::shared_ptr<OpExpr> weight_op_;
};

class UniqueFunctor {
 public:
  UniqueFunctor() {
    op_ = CHECK_JUST(
        OpBuilder("unique").Input("x").Output("y").Output("idx").Output("num_unique").Build());
  };
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& x, const bool sorted,
                           const Symbol<DType>& dtype) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("out_idx", "sorted");
    DataType out_idx = dtype->data_type();
    attrs.SetAllAttrs(out_idx, sorted);
    std::shared_ptr<TensorTuple> output = JUST(
        OpInterpUtil::Dispatch<TensorTuple>(*op_, {JUST(functional::Flatten(x, 0, -1))}, attrs));
    int64_t num_unique = 0;
    std::shared_ptr<Tensor> num_unique_tensor = output->at(2);
    {
      if (num_unique_tensor->is_global()) {
        num_unique_tensor = JUST(GlobalToLocal(num_unique_tensor, false));
      }
      const auto& callback = [&](ep::Stream* stream,
                                 const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
        SyncAutoMemcpy(stream, &num_unique, eager_blob_object->dptr(),
                       GetSizeOfDataType(dtype->data_type()), memory::MakeHostMemCase(),
                       eager_blob_object->mem_case());
      };
      JUST(SyncAccessTensorWithTimeOut(num_unique_tensor, callback, "const"));
    }
    return functional::Slice(output->at(0), /*start=*/{0}, /*end=*/{num_unique}, /*step=*/{1},
                             false);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UniqueWithCountsFunctor {
 public:
  UniqueWithCountsFunctor() {
    unique_op_ = CHECK_JUST(
        OpBuilder("unique").Input("x").Output("y").Output("idx").Output("num_unique").Build());
    unique_with_counts_op_ = CHECK_JUST(OpBuilder("unique_with_counts")
                                            .Input("x")
                                            .Output("y")
                                            .Output("idx")
                                            .Output("num_unique")
                                            .Output("count")
                                            .Build());
  };
  Maybe<TensorTuple> operator()(const std::shared_ptr<Tensor>& x, const bool sorted,
                                const bool return_inverse, const bool return_counts,
                                const Symbol<DType>& dtype) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("out_idx", "sorted");
    attrs.SetAllAttrs(dtype->data_type(), sorted);
    std::shared_ptr<TensorTuple> output;
    if (return_counts) {
      output = JUST(OpInterpUtil::Dispatch<TensorTuple>(
          *unique_with_counts_op_, {JUST(functional::Flatten(x, 0, -1))}, attrs));
    } else {
      output = JUST(OpInterpUtil::Dispatch<TensorTuple>(
          *unique_op_, {JUST(functional::Flatten(x, 0, -1))}, attrs));
    }

    int64_t num_unique = 0;
    std::shared_ptr<Tensor> num_unique_tensor = output->at(2);
    {
      if (num_unique_tensor->is_global()) {
        num_unique_tensor = JUST(GlobalToLocal(num_unique_tensor, false));
      }
      const auto& callback = [&](ep::Stream* stream,
                                 const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
        SyncAutoMemcpy(stream, &num_unique, eager_blob_object->dptr(),
                       GetSizeOfDataType(dtype->data_type()), memory::MakeHostMemCase(),
                       eager_blob_object->mem_case());
      };
      JUST(SyncAccessTensorWithTimeOut(num_unique_tensor, callback, "const"));
    }
    auto result = std::make_shared<TensorTuple>();
    const auto& y = JUST(
        functional::Slice(output->at(0), /*start=*/{0}, /*end=*/{num_unique}, /*step=*/{1}, false));
    result->emplace_back(y);
    if (return_inverse) {
      result->emplace_back(JUST(functional::Reshape(output->at(1), *x->shape())));
    }
    if (return_counts) {
      const auto count = JUST(functional::Slice(output->at(3), /*start=*/{0}, /*end=*/{num_unique},
                                                /*step=*/{1}, false));
      result->emplace_back(count);
    }
    return result;
  }

 private:
  std::shared_ptr<OpExpr> unique_op_;
  std::shared_ptr<OpExpr> unique_with_counts_op_;
};

class BaddBmmFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& batch1,
                           const std::shared_ptr<one::Tensor>& batch2, const double& beta,
                           const double& alpha) const {
    const int32_t batch1_ndim = batch1->ndim();
    const int32_t batch2_ndim = batch2->ndim();
    CHECK_EQ_OR_RETURN(batch1_ndim, 3) << Error::RuntimeError() << "batch1 must be a 3D tensor";
    CHECK_EQ_OR_RETURN(batch2_ndim, 3) << Error::RuntimeError() << "batch2 must be a 3D tensor";
    CHECK_EQ_OR_RETURN(batch1->dim(0), batch2->dim(0))
        << Error::RuntimeError() << "batch1 and batch2 must have same number of batches, got ,"
        << batch1->dim(0) << " and " << batch2->dim(0);
    CHECK_EQ_OR_RETURN(batch1->dim(2), batch2->dim(1))
        << "Incompatible matrix sizes for bmm (" << batch1->dim(1) << "x" << batch1->dim(2)
        << " and " << batch2->dim(1) << "x" << batch2->dim(2) << ")";

    if (beta == 0.0) {
      // In stable diffsion, the beta param is always 0.0, so we can avoid use add and mul op to
      // optimize speed and bandwidth in cuda.
      return JUST(functional::BatchMatMul(batch1, batch2, false, false, alpha));
    } else {
      // TODO(add a fuse kernel to optimize speed and bancwidth in cuda)
      return JUST(
          functional::Add(JUST(functional::ScalarMul(beta, input)),
                          JUST(functional::BatchMatMul(batch1, batch2, false, false, alpha)),
                          /*alpha=*/1.0, /*inplace=*/false));
    }
  }
};

class SortFunctor {
 public:
  Maybe<TensorTuple> operator()(const std::shared_ptr<Tensor>& input, const int32_t& dim,
                                const bool descending) const {
    auto outputs = std::make_shared<TensorTuple>(2);
    std::shared_ptr<Tensor> values;
    std::shared_ptr<Tensor> indices;
    int32_t axis = dim;
    axis = JUST(maybe_wrap_dim(axis, input->ndim()));
    std::string direction("ASCENDING");
    if (descending) { direction.assign("DESCENDING"); }
    if (axis == input->ndim() - 1) {
      indices = JUST(ArgSort(input, direction));
      values = JUST(DimGather(input, axis, indices, false));
    } else {
      std::shared_ptr<std::vector<int32_t>> perm =
          JUST(GetPermWhenTransposeAxisToLastDim(input->ndim(), dim));
      auto x = JUST(Transpose(input, *perm));
      auto indices_temp = JUST(ArgSort(x, direction));
      auto inversed_perm = JUST(GetInversedPerm(*perm));
      indices = JUST(Transpose(indices_temp, *inversed_perm));
      values = JUST(DimGather(input, axis, indices, false));
    }
    (*outputs)[0] = values;
    (*outputs)[1] = indices;
    return outputs;
  }
};

class CloneFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& input) const { return input->clone(); }
};

class FusedCodegeexQkvReshapeFunctor {
 public:
  FusedCodegeexQkvReshapeFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("fused_codegeex_qkv_reshape")
                         .Input("query")
                         .Input("key")
                         .Input("value")
                         .Output("new_query")
                         .Output("new_key")
                         .Output("new_value")
                         .Build());
  }

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& query,
                                const std::shared_ptr<one::Tensor>& key,
                                const std::shared_ptr<one::Tensor>& value,
                                const int32_t num_attention_heads) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("num_attention_heads");
    attrs.SetAllAttrs(num_attention_heads);
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {query, key, value}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::ArgMaxFunctor>("ArgMax");
  m.add_functor<impl::ArgMinFunctor>("ArgMin");
  m.add_functor<impl::GlobalTensorConstantFunctor>("GlobalTensorConstant");
  m.add_functor<impl::TensorConstantFunctor>("TensorConstant");
  m.add_functor<impl::GlobalConstantFunctor>("GlobalConstant");
  m.add_functor<impl::ConstantFunctor>("Constant");
  m.add_functor<impl::GlobalEmptyFunctor>("GlobalEmpty");
  m.add_functor<impl::EmptyFunctor>("Empty");
  m.add_functor<impl::EmptyStridedFunctor>("EmptyStrided");
  m.add_functor<impl::ZerosLikeFunctor>("ZerosLike");
  m.add_functor<impl::OnesLikeFunctor>("OnesLike");
  m.add_functor<impl::FullLikeFunctor>("FullLike");
  m.add_functor<impl::FlattenFunctor>("Flatten");
  m.add_functor<impl::FillFunctor>("Fill");
  m.add_functor<impl::FillTensorFunctor>("FillTensor");
  m.add_functor<impl::WhereFunctor>("Where");
  m.add_functor<impl::WhereScalarXFunctor>("WhereScalarX");
  m.add_functor<impl::WhereScalarYFunctor>("WhereScalarY");
  m.add_functor<impl::WhereScalarXYFunctor>("WhereScalarXY");
  m.add_functor<impl::ArgWhereFunctor>("ArgWhere");
  m.add_functor<impl::NonZeroFunctor>("NonZero");
  m.add_functor<impl::BroadcastLikeFunctor>("BroadcastLike");
  m.add_functor<impl::ConcatFunctor>("Concat");
  m.add_functor<impl::StackFunctor>("Stack");
  m.add_functor<impl::StackGradFunctor>("StackGrad");
  m.add_functor<impl::AtLeast1DFunctor>("AtLeast1D");
  m.add_functor<impl::AtLeast1DListFunctor>("AtLeast1D");
  m.add_functor<impl::AtLeast2DFunctor>("AtLeast2D");
  m.add_functor<impl::AtLeast2DListFunctor>("AtLeast2D");
  m.add_functor<impl::AtLeast3DFunctor>("AtLeast3D");
  m.add_functor<impl::AtLeast3DListFunctor>("AtLeast3D");
  m.add_functor<impl::HStackFunctor>("HStack");
  m.add_functor<impl::ColumnStackFunctor>("ColumnStack");
  m.add_functor<impl::VStackFunctor>("VStack");
  m.add_functor<impl::RowStackFunctor>("RowStack");
  m.add_functor<impl::DStackFunctor>("DStack");
  m.add_functor<impl::ExpandFunctor>("Expand");
  m.add_functor<impl::ExpandDimsFunctor>("ExpandDims");
  m.add_functor<impl::ExpandDimsFunctor>("Unsqueeze");
  m.add_functor<impl::UnsqueezeMultipleFunctor>("UnsqueezeMultiple");
  m.add_functor<impl::InplaceUnsqueezeFunctor>("InplaceUnsqueeze");
  m.add_functor<impl::SqueezeFunctor>("Squeeze");
  m.add_functor<impl::InplaceSqueezeFunctor>("InplaceSqueeze");
  m.add_functor<impl::RollFunctor>("Roll");
  m.add_functor<impl::GatherFunctor>("Gather");
  m.add_functor<impl::DimGatherFunctor>("DimGather");
  m.add_functor<impl::ArgSortFunctor>("ArgSort");
  m.add_functor<impl::SearchSortedFunctor>("SearchSorted");
  m.add_functor<impl::SearchSortedScalarFunctor>("SearchSortedScalar");
  m.add_functor<impl::GatherNdFunctor>("GatherNd");
  m.add_functor<impl::ScatterNdFunctor>("ScatterNd");
  m.add_functor<impl::TensorScatterNdUpdateFunctor>("TensorScatterNdUpdate");
  m.add_functor<impl::ScatterNdLikeFunctor>("ScatterNdLike");
  m.add_functor<impl::ReshapeFunctor>("Reshape");
  m.add_functor<impl::ViewFunctor>("View");
  m.add_functor<impl::ToContiguousFunctor>("ToContiguous");
  m.add_functor<impl::InplaceToContiguousFunctor>("InplaceToContiguous");
  m.add_functor<impl::NarrowFunctor>("Narrow");
  m.add_functor<impl::NarrowGradFunctor>("NarrowGrad");
  m.add_functor<impl::SliceUpdateFunctor>("SliceUpdate");
  m.add_functor<impl::SliceFunctor>("Slice");
  m.add_functor<impl::SliceGradFunctor>("SliceGrad");
  m.add_functor<impl::SliceView1dContiguousFunctor>("SliceView1dContiguous");
  m.add_functor<impl::CopyFunctor, impl::CopyToDeviceFunctor>("Copy");
  m.add_functor<impl::FlipFunctor>("Flip");
  m.add_functor<impl::UnfoldTensorFunctor>("UnfoldTensor");
  m.add_functor<impl::UnfoldTensorGradFunctor>("UnfoldTensorGrad");
  m.add_functor<impl::UpsampleGradFunctor>("UpsampleGrad");
  m.add_functor<impl::UpsampleNearest2DFunctor>("UpsampleNearest2D");
  m.add_functor<impl::UpsampleNearest2DGradFunctor>("UpsampleNearest2DGrad");
  m.add_functor<impl::UpsampleBilinear2DFunctor>("UpsampleBilinear2D");
  m.add_functor<impl::UpsampleBilinear2DGradFunctor>("UpsampleBilinear2DGrad");
  m.add_functor<impl::UpsampleLinear1DFunctor>("UpsampleLinear1D");
  m.add_functor<impl::UpsampleLinear1DGradFunctor>("UpsampleLinear1DGrad");
  m.add_functor<impl::UpsampleNearest1DFunctor>("UpsampleNearest1D");
  m.add_functor<impl::UpsampleNearest1DGradFunctor>("UpsampleNearest1DGrad");
  m.add_functor<impl::UpsampleBicubic2DFunctor>("UpsampleBicubic2D");
  m.add_functor<impl::UpsampleBicubic2DGradFunctor>("UpsampleBicubic2DGrad");
  m.add_functor<impl::UpsampleNearest3DFunctor>("UpsampleNearest3D");
  m.add_functor<impl::UpsampleNearest3DGradFunctor>("UpsampleNearest3DGrad");
  m.add_functor<impl::UpsampleTrilinear3DFunctor>("UpsampleTrilinear3D");
  m.add_functor<impl::UpsampleTrilinear3DGradFunctor>("UpsampleTrilinear3DGrad");
  m.add_functor<impl::UnsortedSegmentSumLikeFunctor>("UnsortedSegmentSumLike");
  m.add_functor<impl::UnsortedSegmentSumFunctor>("UnsortedSegmentSum");
  m.add_functor<impl::TrilFunctor>("Tril");
  m.add_functor<impl::InplaceTrilFunctor>("InplaceTril");
  m.add_functor<impl::TriuFunctor>("Triu");
  m.add_functor<impl::InplaceTriuFunctor>("InplaceTriu");
  m.add_functor<impl::DiagFunctor>("Diag");
  m.add_functor<impl::DiagGradFunctor>("DiagGrad");
  m.add_functor<impl::DiagonalFunctor>("Diagonal");
  m.add_functor<impl::DiagonalGradFunctor>("DiagonalGrad");
  m.add_functor<impl::TensorGetItemFunctor>("TensorGetItem");
  m.add_functor<impl::DimScatterFunctorImpl<impl::DimScatterType::kUpdate>>("DimScatterUpdate");
  m.add_functor<impl::DimScatterFunctorImpl<impl::DimScatterType::kAdd>>("DimScatterAdd");
  m.add_functor<impl::DimScatterFunctorImpl<impl::DimScatterType::kMultiply>>("DimScatterMul");
  m.add_functor<impl::DimScatterFunctor>("DimScatter");
  m.add_functor<impl::DimScatterScalarFunctorImpl<impl::DimScatterType::kUpdate>>(
      "DimScatterUpdateScalar");
  m.add_functor<impl::DimScatterScalarFunctorImpl<impl::DimScatterType::kAdd>>(
      "DimScatterAddScalar");
  m.add_functor<impl::DimScatterScalarFunctorImpl<impl::DimScatterType::kMultiply>>(
      "DimScatterMulScalar");
  m.add_functor<impl::DimScatterScalarFunctor>("DimScatterScalar");
  m.add_functor<impl::DimScatterAddLikeFunctor>("DimScatterAddLike");

  m.add_functor<impl::TensorSetItemFunctor>("TensorSetItem");
  m.add_functor<impl::CastLikeFunctor>("CastLike");
  m.add_functor<impl::ElementwiseMinimumGradFunctor>("ElementwiseMinGrad");
  m.add_functor<impl::ElementwiseMaximumGradFunctor>("ElementwiseMaxGrad");
  m.add_functor<impl::BroadcastPowXGradFunctor>("BroadcastPowXGrad");
  m.add_functor<impl::BroadcastPowYGradFunctor>("BroadcastPowYGrad");
  m.add_functor<impl::DivGradFunctor>("DivGrad");
  m.add_functor<impl::IdentityFunctor>("Identity");
  m.add_functor<impl::AmpWhiteIdentityFunctor>("AmpWhiteIdentity");
  m.add_functor<impl::AmpBlackIdentityFunctor>("AmpBlackIdentity");
  m.add_functor<impl::ReduceSumLikeFunctor>("ReduceSumLike");
  m.add_functor<impl::BroadcastReduceSumLikeFunctor>("BroadcastReduceSumLike");
  m.add_functor<impl::SplitFunctor>("Split");
  m.add_functor<impl::UnbindFunctor>("Unbind");
  m.add_functor<impl::ChunkFunctor>("Chunk");
  m.add_functor<impl::SplitLikeFunctor>("SplitLike");
  m.add_functor<impl::SplitWithSizeFunctor>("SplitWithSize");
  m.add_functor<impl::BatchGatherFunctor>("BatchGather");
  m.add_functor<impl::UnsortedBatchSegmentSumFunctor>("UnsortedBatchSegmentSum");
  m.add_functor<impl::MaskedFillFunctor<false>>("MaskedFill");
  m.add_functor<impl::MaskedFillFunctor<true>>("MaskedFillInplace");
  m.add_functor<impl::MeshgridFunctor>("Meshgrid");
  m.add_functor<impl::IndexSelectFunctor>("IndexSelect");
  m.add_functor<impl::ToFunctor, impl::To2Functor, impl::To3Functor, impl::To4Functor,
                impl::ToDeviceFunctor>("To");
  m.add_functor<impl::TopKFunctor>("TopK");
  m.add_functor<impl::InTopKFunctor>("InTopK");
  m.add_functor<impl::TensorToTensorBufferFunctor>("TensorToTensorBuffer");
  m.add_functor<impl::TensorBufferToTensorFunctor>("TensorBufferToTensor");
  m.add_functor<impl::GenTensorBufferFunctor>("GenTensorBuffer");
  m.add_functor<impl::RepeatFunctor>("Repeat");
  m.add_functor<impl::RepeatInterLeaveIndexFunctor>("RepeatInterLeaveIndex");
  m.add_functor<impl::RepeatInterLeaveIntFunctor>("RepeatInterLeaveInt");
  m.add_functor<impl::RepeatInterLeaveTensorFunctor>("RepeatInterLeaveTensor");
  m.add_functor<impl::TileFunctor>("Tile");
  m.add_functor<impl::TransposeAllDimPropertyFunctor>("TransposeAllDimProperty");
  m.add_functor<impl::TransposeAllDimFunctionFunctor>("TransposeAllDimFunction");
  m.add_functor<impl::ReshapeLikeFunctor>("ReshapeLike");
  m.add_functor<impl::PinMemoryFunctor>("PinMemory");
  m.add_functor<impl::BroadcastShapesFunctor>("BroadcastShapes");
  m.add_functor<impl::BroadcastTensorsFunctor>("BroadcastTensors");
  m.add_functor<impl::ExpandFunctor>("BroadcastTo");  // BroadcastTo is an alias of Expand
  m.add_functor<impl::BinCountFunctor>("BinCount");
  m.add_functor<impl::IndexAddFunctor>("IndexAdd");
  m.add_functor<impl::IndexAddInplaceFunctor>("IndexAddInplace");
  m.add_functor<impl::UniqueFunctor>("Unique");
  m.add_functor<impl::UniqueWithCountsFunctor>("UniqueWithCounts");
  m.add_functor<impl::BaddBmmFunctor>("BaddBmm");
  m.add_functor<impl::SortFunctor>("Sort");
  m.add_functor<impl::CloneFunctor>("Clone");
  m.add_functor<impl::FusedCodegeexQkvReshapeFunctor>("FusedCodegeexQkvReshape");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
