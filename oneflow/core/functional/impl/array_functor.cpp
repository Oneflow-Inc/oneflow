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
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/placement_utils.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/random_generator_impl.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/sequence_function.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/functional/impl/unary_functor.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/ep/include/device_manager_registry.h"
#include "oneflow/api/common/ofblob.h"
#include "oneflow/core/framework/tensor_util.h"
#include "oneflow/core/vm/virtual_machine.h"
#include "oneflow/core/framework/tensor_util.h"
#include "oneflow/core/job/nd_sbp_util.h"

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
    return sequence_function(Negative)
        .then(std::bind(ArgMax, std::placeholders::_1, dim, keepdim, dtype))
        .call(input);
  }
};
class GlobalConstantFunctor {
 public:
  GlobalConstantFunctor() { op_ = CHECK_JUST(one::OpBuilder("constant").Output("out").Build()); }
  Maybe<Tensor> operator()(const Shape& shape, const Scalar& value, const Symbol<DType>& dtype,
                           const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp_tuple) const {
    JUST(CheckDeviceIdsIsValid(placement));
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<Shape>("shape", shape));
    JUST(attrs.SetAttr<DataType>("dtype", dtype->data_type()));
    if (IsIntegralDataType(dtype->data_type())) {
      JUST(attrs.SetAttr<bool>("is_floating_value", false));
      JUST(attrs.SetAttr<int64_t>("integer_value", value.As<int64_t>()));
    } else {
      JUST(attrs.SetAttr<bool>("is_floating_value", true));
      JUST(attrs.SetAttr<double>("floating_value", value.As<double>()));
    }
    if (LazyMode::is_enabled()) {
      std::vector<std::string> nd_sbp(sbp_tuple.size());
      {
        for (int i = 0; i < sbp_tuple.size(); ++i) {
          nd_sbp.at(i) = SbpParallelToString(*sbp_tuple.at(i));
        }
      }
      JUST(attrs.SetAttr<std::vector<std::string>>("nd_sbp", nd_sbp));
    }
    const auto& nd_sbp = JUST(GetNdSbp(sbp_tuple));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {}, OpExprInterpContext(attrs, placement, nd_sbp));
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ConstantFunctor {
 public:
  ConstantFunctor() { op_ = CHECK_JUST(one::OpBuilder("constant").Output("out").Build()); }
  Maybe<Tensor> operator()(const Shape& shape, const Scalar& value, const Symbol<DType>& dtype,
                           const Optional<Symbol<Device>>& device) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<Shape>("shape", shape));
    JUST(attrs.SetAttr<DataType>("dtype", dtype->data_type()));
    if (IsIntegralDataType(dtype->data_type())) {
      JUST(attrs.SetAttr<bool>("is_floating_value", false));
      JUST(attrs.SetAttr<int64_t>("integer_value", value.As<int64_t>()));
    } else {
      JUST(attrs.SetAttr<bool>("is_floating_value", true));
      JUST(attrs.SetAttr<double>("floating_value", value.As<double>()));
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
                           const Optional<Symbol<Device>>& device, const bool pin_memory) const {
    MutableAttrMap attrs;
    Symbol<Device> device_symbol = device.value_or(JUST(Device::New("cpu", 0)));
    JUST(attrs.SetAttr<Shape>("shape", shape));
    JUST(attrs.SetAttr<DataType>("dtype", dtype->data_type()));
    JUST(attrs.SetAttr<bool>("pin_memory", pin_memory));
    JUST(attrs.SetAttr<std::string>("device_type", device_symbol->type()));
    JUST(attrs.SetAttr<int64_t>("device_id", device_symbol->device_id()));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class GlobalEmptyFunctor {
 public:
  GlobalEmptyFunctor() { op_ = CHECK_JUST(one::OpBuilder("empty").Output("out").Build()); }
  Maybe<Tensor> operator()(const Shape& shape, const Symbol<DType>& dtype,
                           const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp_tuple) const {
    JUST(CheckDeviceIdsIsValid(placement));
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<Shape>("shape", shape));
    JUST(attrs.SetAttr<DataType>("dtype", dtype->data_type()));
    if (LazyMode::is_enabled()) {
      std::vector<std::string> nd_sbp(sbp_tuple.size());
      {
        for (int i = 0; i < sbp_tuple.size(); ++i) {
          nd_sbp.at(i) = SbpParallelToString(*sbp_tuple.at(i));
        }
      }
      JUST(attrs.SetAttr<std::vector<std::string>>("nd_sbp", nd_sbp));
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

class FlattenFunctor {
 public:
  FlattenFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("flatten").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const int32_t& start_dim,
                           const int32_t& end_dim) const {
    const auto& x_shape = x->shape();
    const int32_t x_dim = x_shape->dim_vec().size();

    int new_start_dim = start_dim;
    int new_end_dim = end_dim;
    if (start_dim < 0) { new_start_dim += x_dim; }
    if (end_dim < 0) { new_end_dim += x_dim; }
    if (new_start_dim == new_end_dim) { return x; }

    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("start_dim", start_dim));
    JUST(attrs.SetAttr<int32_t>("end_dim", end_dim));

    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
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
  WhereScalarXFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("where_scalar_x").Input("condition").Input("y").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& condition, const Scalar& scalar,
                           const std::shared_ptr<one::Tensor>& y) const {
    MutableAttrMap attrs;
    if (scalar.IsBool()) {
      JUST(attrs.SetAttr<bool>("bool_operand", scalar.As<bool>()));
      JUST(attrs.SetAttr<bool>("has_bool_operand", true));
      JUST(attrs.SetAttr<bool>("has_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_int_operand", false));
    } else if (scalar.IsFloatingPoint()) {
      JUST(attrs.SetAttr<double>("float_operand", scalar.As<double>()));
      JUST(attrs.SetAttr<bool>("has_bool_operand", false));
      JUST(attrs.SetAttr<bool>("has_float_operand", true));
      JUST(attrs.SetAttr<bool>("has_int_operand", false));
    } else if (scalar.IsIntegral()) {
      JUST(attrs.SetAttr<int64_t>("int_operand", scalar.As<int64_t>()));
      JUST(attrs.SetAttr<bool>("has_bool_operand", false));
      JUST(attrs.SetAttr<bool>("has_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_int_operand", true));
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "The scalar in Where shoule be float or int.";
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {condition, y}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class WhereScalarYFunctor {
 public:
  WhereScalarYFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("where_scalar_y").Input("condition").Input("x").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& condition,
                           const std::shared_ptr<one::Tensor>& x, const Scalar& scalar) const {
    MutableAttrMap attrs;
    if (scalar.IsBool()) {
      JUST(attrs.SetAttr<bool>("bool_operand", scalar.As<bool>()));
      JUST(attrs.SetAttr<bool>("has_bool_operand", true));
      JUST(attrs.SetAttr<bool>("has_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_int_operand", false));
    } else if (scalar.IsFloatingPoint()) {
      JUST(attrs.SetAttr<double>("float_operand", scalar.As<double>()));
      JUST(attrs.SetAttr<bool>("has_bool_operand", false));
      JUST(attrs.SetAttr<bool>("has_float_operand", true));
      JUST(attrs.SetAttr<bool>("has_int_operand", false));
    } else if (scalar.IsIntegral()) {
      JUST(attrs.SetAttr<int64_t>("int_operand", scalar.As<int64_t>()));
      JUST(attrs.SetAttr<bool>("has_bool_operand", false));
      JUST(attrs.SetAttr<bool>("has_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_int_operand", true));
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "The scalar in Where shoule be bool, float or int.";
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {condition, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class WhereScalarXYFunctor {
 public:
  WhereScalarXYFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("where_scalar_xy").Input("condition").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& condition, const Scalar& x_scalar,
                           const Scalar& y_scalar) const {
    MutableAttrMap attrs;
    if (x_scalar.IsBool() && y_scalar.IsBool()) {
      JUST(attrs.SetAttr<bool>("x_bool_operand", x_scalar.As<bool>()));
      JUST(attrs.SetAttr<bool>("y_bool_operand", y_scalar.As<bool>()));
      JUST(attrs.SetAttr<bool>("has_x_bool_operand", true));
      JUST(attrs.SetAttr<bool>("has_y_bool_operand", true));
      JUST(attrs.SetAttr<bool>("has_x_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_y_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_x_int_operand", false));
      JUST(attrs.SetAttr<bool>("has_y_int_operand", false));
    } else if (x_scalar.IsFloatingPoint() && y_scalar.IsFloatingPoint()) {
      JUST(attrs.SetAttr<double>("x_float_operand", x_scalar.As<double>()));
      JUST(attrs.SetAttr<double>("y_float_operand", y_scalar.As<double>()));
      JUST(attrs.SetAttr<bool>("has_x_bool_operand", false));
      JUST(attrs.SetAttr<bool>("has_y_bool_operand", false));
      JUST(attrs.SetAttr<bool>("has_x_float_operand", true));
      JUST(attrs.SetAttr<bool>("has_y_float_operand", true));
      JUST(attrs.SetAttr<bool>("has_x_int_operand", false));
      JUST(attrs.SetAttr<bool>("has_y_int_operand", false));
    } else if (x_scalar.IsIntegral() && y_scalar.IsIntegral()) {
      JUST(attrs.SetAttr<int64_t>("x_int_operand", x_scalar.As<int64_t>()));
      JUST(attrs.SetAttr<int64_t>("y_int_operand", y_scalar.As<int64_t>()));
      JUST(attrs.SetAttr<bool>("has_x_bool_operand", false));
      JUST(attrs.SetAttr<bool>("has_y_bool_operand", false));
      JUST(attrs.SetAttr<bool>("has_x_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_y_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_x_int_operand", true));
      JUST(attrs.SetAttr<bool>("has_y_int_operand", true));
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "The scalar in Where shoule be bool, float or int.";
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {condition}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ArgWhereFunctor {
 public:
  ArgWhereFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("argwhere").Input("input").Output("output").Output("output_size").Build());
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& x,
                                const Symbol<DType>& dtype) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<DataType>("dtype", dtype->data_type()));
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
      JUST(CopyLocalTensorDataTo(size->is_local() ? size : JUST(size->cur_rank_phy_tensor()),
                                 (void*)(&size_val), GetSizeOfDataType(DataType::kInt64)));
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
    MutableAttrMap attrs;
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
      JUST(attrs.SetAttr<std::vector<int32_t>>("broadcast_axes", broadcast_axes));
    } else {
      JUST(attrs.SetAttr<std::vector<int32_t>>("broadcast_axes", broadcast_axes));
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
    int64_t max_dim_size = 0;
    CHECK_GE_OR_RETURN(ninput, 1) << Error::RuntimeError() << "inputs size must greater than 0";
    axis = JUST(maybe_wrap_dim(axis, ndim));

    const std::shared_ptr<const Shape>& shape = inputs[0]->shape();
    for (const auto& input : inputs) {
      CHECK_OR_RETURN(input->ndim() == ndim)
          << Error::RuntimeError() << "Tensors must have same number of dimensions: got "
          << input->ndim() << " and " << ndim << " is expected.";
      for (int i = 0; i < ndim; ++i) {
        if (axis == i) {
          max_dim_size += input->shape()->At(i);
        } else {
          CHECK_OR_RETURN(input->shape()->At(i) == shape->At(i))
              << Error::RuntimeError() << "Sizes of tensors must match except in dimension " << axis
              << ". Got " << input->shape()->At(i) << " and " << shape->At(i)
              << " is expected in dimension 1.";
        }
      }
    }

    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("axis", axis));
    JUST(attrs.SetAttr<int64_t>("max_dim_size", max_dim_size));
    TensorTuple outputs;
    for (int i = 0; i < ninput; i += kMaxInputCount) {
      size_t size = (i + kMaxInputCount) < ninput ? kMaxInputCount : ninput - i;
      TensorTuple partial_inputs(size);
      TensorProcessor tensor_processor;
      for (int j = 0; j < size; ++j) { partial_inputs[j] = inputs[i + j]; }
      JUST(tensor_processor.PromoteInputsToCommonDtype(true).AddInputs(partial_inputs).Apply());
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
    if (ninput == 1) { return ExpandDims(inputs[0], dim); }
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("axis", stack_dim));
    JUST(attrs.SetAttr<int64_t>("max_dim_size", max_dim_size));
    TensorTuple outputs;
    for (int i = 0; i < ninput; i += kMaxInputCount) {
      size_t size = (i + kMaxInputCount) < ninput ? kMaxInputCount : ninput - i;
      TensorTuple partial_inputs(size);
      for (int j = 0; j < size; ++j) { partial_inputs[j] = inputs[i + j]; }
      outputs.emplace_back(
          JUST(OpInterpUtil::Dispatch<Tensor>(*ops_.at(size - 1), partial_inputs, attrs)));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("axis", axis));
    TensorTuple inputs(like.size() + 1);
    inputs[0] = x;
    for (int i = 0; i < like.size(); ++i) { inputs[i + 1] = like[i]; }
    return OpInterpUtil::Dispatch<TensorTuple>(*ops_.at(like.size() - 1), inputs, attrs);
  }

 private:
  std::vector<std::shared_ptr<OpExpr>> ops_;
};

class ExpandFunctor {
 public:
  ExpandFunctor() { op_ = CHECK_JUST(one::OpBuilder("expand").Input("in").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Shape& shape) const {
    CHECK_GE_OR_RETURN(shape.NumAxes(), x->shape()->NumAxes())
        << Error::RuntimeError() << "expand(tensor{" << x->shape()->ToString()
        << "}, size=" << x->shape()->NumAxes() << "): the number of sizes provided ("
        << shape.NumAxes() << ") "
        << "must be greater or equal to the number of dimensions in the tensor ("
        << x->shape()->NumAxes() << ")";
    std::vector<int32_t> in_shape(x->shape()->NumAxes());
    for (int i = 0; i < in_shape.size(); ++i) { in_shape[i] = x->shape()->At(i); }

    // check the parameters
    int shift = shape.NumAxes() - in_shape.size();
    for (int i = shape.NumAxes() - 1; i >= 0; --i) {
      int index = i - shift;
      if (index >= 0) {
        if (shape.At(i) != -1 && shape.At(i) != in_shape[index]) {
          CHECK_OR_RETURN(shape.At(i) >= 0 && in_shape[index] == 1)
              << Error::RuntimeError() << "The expanded size of the tensor (" << shape.At(i)
              << ") must match the existing size (" << in_shape[index]
              << ") at non-singleton dimension " << i << ".  Target sizes: " << shape.ToString()
              << ".  Tensor sizes: " << x->shape()->ToString();
        }
      } else {
        CHECK_GE_OR_RETURN(shape.At(i), 0)
            << Error::RuntimeError() << "The expanded size of the tensor (" << shape.At(i)
            << ") isn't allowed in a leading, non-existing dimension " << i
            << " .Target size: " << shape.ToString();
      }
    }

    std::vector<int32_t> expand_shape(shape.NumAxes());
    for (int i = 0; i < shape.NumAxes(); ++i) { expand_shape[i] = shape.dim_vec().at(i); }

    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int32_t>>("logical_in_shape", in_shape));
    JUST(attrs.SetAttr<std::vector<int32_t>>("logical_expand_shape", expand_shape));

    // if input tensor is eager local, then try return tensor's view
    if (view::IsViewApplicable(x)) { return view::Expand(x, in_shape, expand_shape); }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ExpandGradFunctor {
 public:
  ExpandGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("expand_grad").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::vector<int32_t>& logical_in_shape,
                           const std::vector<int32_t>& logical_expand_shape) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int32_t>>("logical_out_shape", logical_in_shape));
    JUST(attrs.SetAttr<std::vector<int32_t>>("logical_expand_shape", logical_expand_shape));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy}, attrs);
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
    JUST(maybe_wrap_dim(dim, ndim + 1));
    if (dim < 0) { expand_dim = dim + ndim + 1; }
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("axis", expand_dim));

    if (view::IsViewApplicable(input)) { return view::Unsqueeze(input, expand_dim); }

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

    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int32_t>>("axes", squeeze_dims));

    if (view::IsViewApplicable(x)) { return view::Squeeze(x, squeeze_dims); }

    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class RollFunctor {
 public:
  RollFunctor() { op_ = CHECK_JUST(one::OpBuilder("roll").Input("in").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::vector<int32_t>& shifts,
                           const Optional<std::vector<int32_t>>& dims) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int32_t>>("shifts", shifts));

    std::vector<int32_t> actual_dims;
    if (dims.has_value()) {
      actual_dims = *JUST(dims);
    } else {
      actual_dims.emplace_back(-1);
    }
    CHECK_EQ_OR_RETURN(shifts.size(), actual_dims.size())
        << Error::RuntimeError() << "shifts and dimensions must align. shifts: " << shifts.size()
        << ", dims: " << actual_dims.size();
    JUST(attrs.SetAttr<std::vector<int32_t>>("dims", actual_dims));

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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("axis", axis));
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

    JUST(maybe_wrap_dim(dim, index->ndim()));
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
        if (i != dim) {
          CHECK_LE_OR_RETURN(index->shape()->At(i), input->shape()->At(i))
              << Error::RuntimeError() << "Size does not match at dimension " << i
              << " expected index " << *(index->shape()) << " to be smaller than self "
              << *(input->shape()) << " apart from dimension " << dim;
        }
      }
    }

    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("dim", dim));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input, index}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DimScatterFunctor {
 public:
  DimScatterFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("dim_scatter_update")
                         .Input("input")
                         .Input("index")
                         .Input("src")
                         .Output("output")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int32_t& dim,
                           const std::shared_ptr<one::Tensor>& index,
                           const std::shared_ptr<one::Tensor>& src) const {
    MutableAttrMap attrs;
    const int32_t ndim = input->shape()->NumAxes();
    JUST(attrs.SetAttr<int32_t>("dim", dim < 0 ? dim + ndim : dim));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input, index, src}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DimScatterAddFunctor {
 public:
  DimScatterAddFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("dim_scatter_add")
                         .Input("input")
                         .Input("index")
                         .Input("src")
                         .Output("output")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int32_t& dim,
                           const std::shared_ptr<one::Tensor>& index,
                           const std::shared_ptr<one::Tensor>& src) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("dim", dim));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input, index, src}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("dim", dim));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {like, index, src}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DimScatterMulFunctor {
 public:
  DimScatterMulFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("dim_scatter_mul")
                         .Input("input")
                         .Input("index")
                         .Input("src")
                         .Output("output")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int32_t& dim,
                           const std::shared_ptr<one::Tensor>& index,
                           const std::shared_ptr<one::Tensor>& src) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("dim", dim));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input, index, src}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DimScatterUpdateScalarFunctor {
 public:
  DimScatterUpdateScalarFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("dim_scatter_update_scalar")
                         .Input("input")
                         .Input("index")
                         .Output("output")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int32_t& dim,
                           const std::shared_ptr<one::Tensor>& index, const Scalar& src) const {
    MutableAttrMap attrs;
    const int32_t ndim = input->shape()->NumAxes();
    JUST(attrs.SetAttr<int32_t>("dim", dim < 0 ? dim + ndim : dim));
    JUST(attrs.SetAttr<float>("src_scalar", src.As<float>()));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input, index}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DimScatterAddScalarFunctor {
 public:
  DimScatterAddScalarFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("dim_scatter_add_scalar")
                         .Input("input")
                         .Input("index")
                         .Output("output")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int32_t& dim,
                           const std::shared_ptr<one::Tensor>& index, const Scalar& src) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("dim", dim));
    JUST(attrs.SetAttr<float>("src_scalar", src.As<float>()));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input, index}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DimScatterMulScalarFunctor {
 public:
  DimScatterMulScalarFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("dim_scatter_mul_scalar")
                         .Input("input")
                         .Input("index")
                         .Output("output")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int32_t& dim,
                           const std::shared_ptr<one::Tensor>& index, const Scalar& src) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("dim", dim));
    JUST(attrs.SetAttr<float>("src_scalar", src.As<float>()));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input, index}, attrs);
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::string>("direction", direction));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<bool>("out_int32", out_int32));
    JUST(attrs.SetAttr<bool>("right", right));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<bool>("out_int32", out_int32));
    JUST(attrs.SetAttr<bool>("right", right));
    bool is_values_float = values.IsFloatingPoint();
    if (is_values_float) {
      double_t values_tmp = values.As<double_t>();
      JUST(attrs.SetAttr<double>("values", values_tmp));
    } else {
      int64_t values_tmp = values.As<int64_t>();
      JUST(attrs.SetAttr<double>("values", values_tmp));
    }
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<Shape>("shape", shape));
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
    if (inplace) {
      JUST(CheckInplaceValid(tensor));
      auto outputs = std::make_shared<TensorTuple>(1);
      outputs->at(0) = tensor;
      JUST(OpInterpUtil::Dispatch(*op_, {tensor, indices, updates}, outputs.get()));
      return outputs->at(0);
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*op_, {tensor, indices, updates});
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
    Shape infered_shape = *JUST(InferShape(x, shape));

    if (view::IsViewApplicable(x)) {
      Optional<Stride> infered_stride =
          ComputeStride(*(x->shape()), *JUST(x->stride()), infered_shape);
      if (infered_stride.has_value()) {
        return view::Reshape(x, infered_shape, *JUST(infered_stride));
      }
    }
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<Shape>("shape", infered_shape));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ViewFunctor {
 public:
  ViewFunctor() { op_ = CHECK_JUST(one::OpBuilder("reshape").Input("in").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Shape& shape) const {
    Shape infered_shape = *JUST(InferShape(x, shape));
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<Shape>("shape", infered_shape));

    if (view::IsViewApplicable(x)) {
      Optional<Stride> infered_stride =
          ComputeStride(*(x->shape()), *JUST(x->stride()), infered_shape);
      CHECK_OR_RETURN_ERROR(infered_stride.has_value())
          << Error::RuntimeError()
          << "view size is not compatible with input tensor's size and stride (at least one "
             "dimension spans across two contiguous subspaces). Use .reshape(...) instead.";
      return view::Reshape(x, infered_shape, *JUST(infered_stride));
    }

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
    std::shared_ptr<Stride> stride(new Stride(*input->shape()));
    // update stride
    const auto& blob_object = JUST(input->eager_blob_object());
    Symbol<LocalTensorMeta> old_tensor_meta = JUST(input->local_tensor_meta());

    Symbol<LocalTensorMeta> new_tensor_meta = SymbolOf(LocalTensorMeta(
        std::make_shared<Shape>(old_tensor_meta->shape()), stride, old_tensor_meta->dtype(),
        old_tensor_meta->device(), old_tensor_meta->storage_offset()));

    std::shared_ptr<EagerLocalTensorImpl> final_tensor_impl =
        std::make_shared<EagerLocalTensorImpl>(JUST(input->tensor_storage()),
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("dim", narrow_dim));
    JUST(attrs.SetAttr<int64_t>("start", start));
    JUST(attrs.SetAttr<int64_t>("length", length));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("dim", dim));
    JUST(attrs.SetAttr<int64_t>("start", start));
    JUST(attrs.SetAttr<int64_t>("length", length));
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

    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int64_t>>("start", start));
    JUST(attrs.SetAttr<std::vector<int64_t>>("stop", stop));
    JUST(attrs.SetAttr<std::vector<int64_t>>("step", step));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int64_t>>("start", start));
    JUST(attrs.SetAttr<std::vector<int64_t>>("stop", stop));
    JUST(attrs.SetAttr<std::vector<int64_t>>("step", step));

    if (inplace) {
      auto outputs = std::make_shared<TensorTuple>(1);
      JUST(CheckInplaceValid(ref));
      JUST(VectorAt(*outputs, 0)) = ref;
      JUST(OpInterpUtil::Dispatch(*op_, {ref, value}, outputs.get(), attrs));
      return JUST(VectorAt(*outputs, 0));
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*op_, {ref, value}, attrs);
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<Shape>("like_shape", like_shape));
    JUST(attrs.SetAttr<std::vector<int64_t>>("start", start));
    JUST(attrs.SetAttr<std::vector<int64_t>>("stop", stop));
    JUST(attrs.SetAttr<std::vector<int64_t>>("step", step));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("height_scale", height_scale));
    JUST(attrs.SetAttr<double>("width_scale", width_scale));
    JUST(attrs.SetAttr<bool>("align_corners", align_corners));
    JUST(attrs.SetAttr<std::string>("interpolation", interpolation));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class CopyFunctor {
 public:
  CopyFunctor() { op_ = CHECK_JUST(one::OpBuilder("copy").Input("in").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const std::string& device_type,
                           const int64_t& device_id, const bool pin_memory) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::string>("device_type", device_type));
    JUST(attrs.SetAttr<int64_t>("device_id", device_id));
    JUST(attrs.SetAttr<bool>("pin_memory", pin_memory));

#ifdef WITH_CUDA
    if (device_type == "cuda") { InitCudaContextOnce(device_id); }
#endif
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class FlipFunctor {
 public:
  FlipFunctor() { op_ = CHECK_JUST(one::OpBuilder("flip").Input("x").Output("y").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::vector<int32_t>& dims) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int32_t>>("dims", dims));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("dimension", dimension));
    JUST(attrs.SetAttr<int32_t>("size", size));
    JUST(attrs.SetAttr<int32_t>("step", step));
    // if input tensor is eager local, than try return tensor's view
    if (view::IsViewApplicable(x)) { return view::UnfoldTensor(x, dimension, size, step); }
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("dimension", dimension));
    JUST(attrs.SetAttr<int32_t>("size", size));
    JUST(attrs.SetAttr<int32_t>("step", step));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("scale_factor", scale_factor));
    JUST(attrs.SetAttr<bool>("align_corners", align_corners));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    if (output_size.has_value()) {
      JUST(attrs.SetAttr<std::vector<int64_t>>("output_size", *JUST(output_size)));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("scale_factor", scale_factor));
    JUST(attrs.SetAttr<bool>("align_corners", align_corners));
    if (output_size.has_value()) {
      JUST(attrs.SetAttr<std::vector<int64_t>>("output_size", *JUST(output_size)));
    }
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("scale_factor", scale_factor));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    if (output_size.has_value()) {
      JUST(attrs.SetAttr<std::vector<int64_t>>("output_size", *JUST(output_size)));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("scale_factor", scale_factor));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    if (output_size.has_value()) {
      JUST(attrs.SetAttr<std::vector<int64_t>>("output_size", *JUST(output_size)));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("height_scale", height_scale));
    JUST(attrs.SetAttr<double>("width_scale", width_scale));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    if (output_size.has_value()) {
      JUST(attrs.SetAttr<std::vector<int64_t>>("output_size", *JUST(output_size)));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("height_scale", height_scale));
    JUST(attrs.SetAttr<double>("width_scale", width_scale));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    if (output_size.has_value()) {
      JUST(attrs.SetAttr<std::vector<int64_t>>("output_size", *JUST(output_size)));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("height_scale", height_scale));
    JUST(attrs.SetAttr<double>("width_scale", width_scale));
    JUST(attrs.SetAttr<bool>("align_corners", align_corners));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    if (output_size.has_value()) {
      JUST(attrs.SetAttr<std::vector<int64_t>>("output_size", *JUST(output_size)));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("height_scale", height_scale));
    JUST(attrs.SetAttr<double>("width_scale", width_scale));
    JUST(attrs.SetAttr<bool>("align_corners", align_corners));
    if (output_size.has_value()) {
      JUST(attrs.SetAttr<std::vector<int64_t>>("output_size", *JUST(output_size)));
    }
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("height_scale", height_scale));
    JUST(attrs.SetAttr<double>("width_scale", width_scale));
    JUST(attrs.SetAttr<bool>("align_corners", align_corners));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    if (output_size.has_value()) {
      JUST(attrs.SetAttr<std::vector<int64_t>>("output_size", *JUST(output_size)));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("height_scale", height_scale));
    JUST(attrs.SetAttr<double>("width_scale", width_scale));
    JUST(attrs.SetAttr<bool>("align_corners", align_corners));
    if (output_size.has_value()) {
      JUST(attrs.SetAttr<std::vector<int64_t>>("output_size", *JUST(output_size)));
    }
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("depth_scale", depth_scale));
    JUST(attrs.SetAttr<double>("height_scale", height_scale));
    JUST(attrs.SetAttr<double>("width_scale", width_scale));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    if (output_size.has_value()) {
      JUST(attrs.SetAttr<std::vector<int64_t>>("output_size", *JUST(output_size)));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("depth_scale", depth_scale));
    JUST(attrs.SetAttr<double>("height_scale", height_scale));
    JUST(attrs.SetAttr<double>("width_scale", width_scale));
    if (output_size.has_value()) {
      JUST(attrs.SetAttr<std::vector<int64_t>>("output_size", *JUST(output_size)));
    }
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("depth_scale", depth_scale));
    JUST(attrs.SetAttr<double>("height_scale", height_scale));
    JUST(attrs.SetAttr<double>("width_scale", width_scale));
    JUST(attrs.SetAttr<bool>("align_corners", align_corners));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    if (output_size.has_value()) {
      JUST(attrs.SetAttr<std::vector<int64_t>>("output_size", *JUST(output_size)));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("depth_scale", depth_scale));
    JUST(attrs.SetAttr<double>("height_scale", height_scale));
    JUST(attrs.SetAttr<double>("width_scale", width_scale));
    JUST(attrs.SetAttr<bool>("align_corners", align_corners));
    if (output_size.has_value()) {
      JUST(attrs.SetAttr<std::vector<int64_t>>("output_size", *JUST(output_size)));
    }
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("axis", axis));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, segment_ids, like}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class TrilFunctor {
 public:
  TrilFunctor() { op_ = CHECK_JUST(one::OpBuilder("tril").Input("in").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const int64_t& diagonal) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("diagonal", diagonal));
    JUST(attrs.SetAttr<bool>("is_floating_fill_value", false));
    JUST(attrs.SetAttr<int64_t>("integer_fill_value", 0));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class TriuFunctor {
 public:
  TriuFunctor() { op_ = CHECK_JUST(one::OpBuilder("triu").Input("in").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const int64_t& diagonal) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("diagonal", diagonal));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DiagFunctor {
 public:
  DiagFunctor() { op_ = CHECK_JUST(one::OpBuilder("diag").Input("in").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const int32_t& diagonal) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("diagonal", diagonal));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("diagonal", diagonal));
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

    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("offset", offset));

    if (view::IsViewApplicable(x)) {
      return view::Diagonal(x, offset, p_dim1, p_dim2);
    } else {
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("offset", offset));
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
      JUST(UnifyLocalTensorAndIndicesOnDevice(x, tensor_indices));
      result = JUST(ApplyAdvancedIndexing(result, tensor_indices));
    }

    // TODO(): Returns a view of tensor `x`.
    if (result == x) { result = JUST(Identity(x)); }
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
    if (expand_dims.size()) {
      slice_indices = *JUST(RemoveExpandDimSlice(slice_indices, expand_dims));
    }
    int64_t ndims = x->shape()->NumAxes();
    CHECK_EQ_OR_RETURN(slice_indices.size(), ndims)
        << Error::RuntimeError() << "Failed to prepare slice indices.";
    // Not support combined indexing now
    if (!tensor_indices.empty()) {
      CHECK_OR_RETURN(tensor_indices.size() == ndims
                      && std::all_of(tensor_indices.begin(), tensor_indices.end(),
                                     [](const std::shared_ptr<Tensor>& index) { return index; }))
          << Error::RuntimeError()
          << "Combining indexing is not support for tensor setitem currently";
    }

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
      if (tensor->ndim() == 0) { tensor = JUST(functional::Reshape(tensor, Shape({1}))); }
    }
    if (tensor_indices.size() == ndims) {  // advance indexing
      if (ndims == 0 && index[0].IsEllipsis()) {
        // for scalar input tensor setitem, only support ellipsis indexing type
        Shape tmp_shape{1};
        const auto& value_tensor = JUST(functional::View(value, tmp_shape));
        const auto& input_tensor = JUST(functional::View(x, tmp_shape));
        std::vector<int64_t> starts(1, 0);
        std::vector<int64_t> stops(1, 1);
        std::vector<int64_t> steps(1, 1);
        JUST(SliceUpdate(input_tensor, value_tensor, starts, stops, steps, /*inplace=*/true));
      } else {
        // advance indexing
        std::shared_ptr<Tensor> indices = JUST(functional::Stack(tensor_indices, 0));
        if (indices->shape()->elem_cnt() == 0) { return Maybe<void>::Ok(); }
        indices = JUST(functional::Transpose(indices, {1, 0}));
        value_tensor = JUST(functional::Expand(value_tensor, {indices->shape()->At(0)}));
        JUST(functional::TensorScatterNdUpdate(x, indices, value_tensor, /*inplace=*/true));
      }
    } else {                              // slice update
      if (target_shape.NumAxes() != 0 &&  // NOLINT
          /*need_expand=*/value_shape->Count(0) != target_shape.Count(0)) {
        // Remove the beginning redundant 1-dimensions.
        if (value_shape->NumAxes() > target_shape.NumAxes()) {
          int64_t start_axis = value_shape->NumAxes() - target_shape.NumAxes();
          const auto& shape = JUST(value_shape->Slice(start_axis, value_shape->NumAxes()));
          value_tensor = JUST(Reshape(value, *shape));
        }
        value_tensor = JUST(Expand(value_tensor, target_shape));
      }
      std::vector<int64_t> start(ndims), end(ndims), step(ndims);
      DimVector slice_dims(ndims);
      for (int i = 0; i < ndims; ++i) {
        const auto& slice = slice_indices.at(i);
        start[i] = slice.start();
        end[i] = slice.end();
        step[i] = slice.step();
        slice_dims[i] = (end[i] - start[i] + step[i] - 1) / step[i];
      }
      Shape slice_shape(slice_dims);
      if (slice_shape != *(value_tensor->shape())) {
        value_tensor = JUST(Reshape(value_tensor, slice_shape));
      }
      JUST(SliceUpdate(x, value_tensor, start, end, step, /*inplace=*/true));
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
  BroadcastPowXGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("broadcast_pow_x_grad")
                         .Input("dz")
                         .Input("x")
                         .Input("y")
                         .Input("z")
                         .Output("dx")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dz,
                           const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y,
                           const std::shared_ptr<one::Tensor>& z) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dz, x, y, z});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class BroadcastPowYGradFunctor {
 public:
  BroadcastPowYGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("broadcast_pow_y_grad")
                         .Input("dz")
                         .Input("x")
                         .Input("y")
                         .Input("z")
                         .Output("dy")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dz,
                           const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y,
                           const std::shared_ptr<one::Tensor>& z) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dz, x, y, z});
  }

 private:
  std::shared_ptr<OpExpr> op_;
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

class ReduceSumLikeFunctor {
 public:
  ReduceSumLikeFunctor() {
    op_ =
        CHECK_JUST(one::OpBuilder("reduce_sum_like").Input("x").Input("like").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& like,
                           const std::vector<int32_t>& axis) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int32_t>>("axis", axis));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, like}, attrs);
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("axis", axis));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("num_segments", num_segments));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {data, segment_ids}, attrs);
  }

 protected:
  std::shared_ptr<OpExpr> op_;
};

class MaskedFillFunctor {
 public:
  MaskedFillFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("masked_fill").Input("x").Input("mask").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& mask, const Scalar& value) const {
    MutableAttrMap attrs;
    if (IsFloatingDataType(x->dtype()->data_type())) {
      JUST(attrs.SetAttr<double>("float_operand", value.As<double>()));
      JUST(attrs.SetAttr<bool>("has_float_operand", true));
      JUST(attrs.SetAttr<bool>("has_int_operand", false));
      JUST(attrs.SetAttr<bool>("has_bool_operand", false));
    } else if (IsIntegralDataType(x->dtype()->data_type())) {
      JUST(attrs.SetAttr<int64_t>("int_operand", value.As<int64_t>()));
      JUST(attrs.SetAttr<bool>("has_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_int_operand", true));
      JUST(attrs.SetAttr<bool>("has_bool_operand", false));
    } else if (IsBoolDataType(x->dtype()->data_type())) {
      JUST(attrs.SetAttr<bool>("bool_operand", value.As<bool>()));
      JUST(attrs.SetAttr<bool>("has_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_int_operand", false));
      JUST(attrs.SetAttr<bool>("has_bool_operand", true));
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "Only support floating or integral data type.";
    }
    const auto& x_shape = *(x->shape());
    const auto& mask_shape = *(mask->shape());
    if (x_shape != mask_shape) {
      Shape max_shape = Shape::Ones(std::max(x_shape.NumAxes(), mask_shape.NumAxes()));
      const Shape& x_extend_shape =
          CreateLeftExtendedShape(ShapeView(x_shape), max_shape.NumAxes());
      const Shape& mask_extend_shape =
          CreateLeftExtendedShape(ShapeView(mask_shape), max_shape.NumAxes());
      FOR_RANGE(int64_t, i, 0, max_shape.NumAxes()) {
        max_shape.Set(i, std::max(x_extend_shape.At(i), mask_extend_shape.At(i)));
      }
      return OpInterpUtil::Dispatch<Tensor>(
          *op_, {JUST(Expand(x, max_shape)), JUST(Expand(mask, max_shape))}, attrs);
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, mask}, attrs);
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
inline Maybe<bool> device_equal(const std::string& device_name, const int device_id,
                                Symbol<Device> device) {
  return (device_name == device->type() && device_id == device->device_id());
}

Maybe<Tensor> LocalTensorTo(const std::shared_ptr<Tensor>& x, const std::string& device_name,
                            const int device_id, const Symbol<DType>& dtype, const bool& copy) {
  std::shared_ptr<Tensor> tensor = x;
  if (!JUST(device_equal(device_name, device_id, JUST(x->device())))) {
    tensor = JUST(Copy(tensor, device_name, device_id, /*pin_memory=*/false));
  }
  if (dtype != x->dtype()) { tensor = JUST(Cast(tensor, dtype, /*pin_memory=*/false)); }
  if (copy && tensor == x) {
    tensor = JUST(Copy(tensor, device_name, device_id, /*pin_memory=*/false));
  }
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
    tensor = JUST(GlobalToLocal(x));
    Symbol<Device> device = JUST(Device::New(device_type));
    tensor = JUST(LocalTensorTo(tensor, device->type(), device->device_id(), dtype, copy));
    JUST(tensor->set_requires_grad(x->requires_grad()));
    return JUST(LocalToGlobal(tensor, placement, sbp_tuple, *(x->shape()), dtype,
                              /* sync_data */ true));
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
      std::string device_name = "";
      int device_id = 0;
      if (device_.has_value()) {
        JUST(ParsingDeviceTag(device_.value_or(""), &device_name, &device_id));
        if (device_id == -1) { device_id = GlobalProcessCtx::LocalRank(); }
      } else {
        Symbol<Device> device = JUST(input->device());
        device_name = device->type();
        device_id = device->device_id();
      }
      return JUST(LocalTensorTo(input, device_name, device_id, dtype, copy));
    }
  }
};

class To2Functor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& input,
                           const Optional<Symbol<Device>>& device_,
                           const Optional<Symbol<DType>>& dtype_, bool copy) const {
    CHECK_OR_RETURN(!(input->is_global() && device_.has_value()))
        << Error::RuntimeError()
        << "Only string device without device id (eg. \"cpu\" or \"cuda\") is expected "
        << "for global tensor, but got " << device_.value_or(Symbol<Device>())->ToRepr();
    if (input->is_global()) {
      std::string device_type = JUST(input->parallel_desc())->device_tag();
      return JUST(GlobalTensorTo(input, device_type, dtype_.value_or(input->dtype()), copy));
    } else {
      auto dtype = dtype_.value_or(input->dtype());
      auto device =
          device_.has_value() ? device_.value_or(Symbol<Device>()) : JUST(input->device());
      return JUST(LocalTensorTo(input, device->type(), device->device_id(), dtype, copy));
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
      return LocalTensorTo(input, device->type(), device->device_id(), dtype, copy);
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
    std::string device_name = device->type();
    int device_id = device->device_id();
    return LocalTensorTo(input, device_name, device_id, dtype, copy);
  }
};

class TopKFunctor {
 public:
  TopKFunctor() { op_ = CHECK_JUST(one::OpBuilder("top_k").Input("in").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& input, int32_t k, bool sorted) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("k", k));
    JUST(attrs.SetAttr<bool>("sorted", sorted));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input}, attrs);
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("k", k));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<Shape>("instance_shape", instance_shape));
    JUST(attrs.SetAttr<DataType>("dtype", dtype->data_type()));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("instance_dims", instance_dims));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<Shape>("shape", shape));
    JUST(attrs.SetAttr<std::vector<Shape>>("shape_list", shape_list));
    JUST(attrs.SetAttr<std::vector<float>>("value_list", value_list));
    JUST(attrs.SetAttr<DataType>("data_type", dtype->data_type()));
    JUST(attrs.SetAttr<bool>("dynamic_out", dynamic_out));
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
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::int64_t>("repeat_num", repeat_num));
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
      const auto& callback = [&](uint64_t ofblob_ptr) {
        CHECK_JUST(BlobBufferCopyUtil<int64_t>::To(ofblob_ptr, repeats_value.data(),
                                                   repeats_value.size()));
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

    auto empty = JUST(functional::Empty(*shape.get(), input->dtype(), device, /*pin_memory=*/true));
    // TODO: remove this requires_grad
    JUST(empty->set_requires_grad(requires_grad));
    const int32_t ndim = input->ndim();
    if (ndim == 0) {
      // for 0-dim tensor
      TensorIndex tensor_index;
      tensor_index.emplace_back(functional::detail::IndexItem(functional::detail::EllipsisIndex{}));
      JUST(functional::TensorSetItem(empty, tensor_index, input));
      return empty;
    } else {
      MutableAttrMap attrs;
      std::vector<int64_t> starts(ndim, 0);
      std::vector<int64_t> stops(ndim);
      std::vector<int64_t> steps(ndim, 1);
      for (int i = 0; i < ndim; ++i) { stops[i] = input->shape()->At(i); }
      JUST(attrs.SetAttr<std::vector<int64_t>>("start", starts));
      JUST(attrs.SetAttr<std::vector<int64_t>>("stop", stops));
      JUST(attrs.SetAttr<std::vector<int64_t>>("step", steps));
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
    MutableAttrMap attrs;
    if (IsFloatingDataType(in->dtype()->data_type())) {
      JUST(attrs.SetAttr<double>("floating_value", value.As<double>()));
      JUST(attrs.SetAttr<bool>("is_floating_value", true));
    } else if (IsIntegralDataType(in->dtype()->data_type())) {
      JUST(attrs.SetAttr<int64_t>("integral_value", value.As<int64_t>()));
      JUST(attrs.SetAttr<bool>("is_floating_value", false));
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
    JUST(tensor_processor.PromoteInputsToCommonDtype(true).AddInputs({in, value}).Apply());
    TensorTuple input_tuple = JUST(tensor_processor.GetInputs());
    auto outputs = std::make_shared<TensorTuple>(1);
    (*outputs)[0] = in;
    JUST(OpInterpUtil::Dispatch(*op_, {input_tuple[0], input_tuple[1]}, outputs.get()));
    return (*outputs)[0];
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::ArgMaxFunctor>("ArgMax");
  m.add_functor<impl::ArgMinFunctor>("ArgMin");
  m.add_functor<impl::GlobalConstantFunctor>("GlobalConstant");
  m.add_functor<impl::ConstantFunctor>("Constant");
  m.add_functor<impl::GlobalEmptyFunctor>("GlobalEmpty");
  m.add_functor<impl::EmptyFunctor>("Empty");
  m.add_functor<impl::ZerosLikeFunctor>("ZerosLike");
  m.add_functor<impl::OnesLikeFunctor>("OnesLike");
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
  m.add_functor<impl::ExpandFunctor>("Expand");
  m.add_functor<impl::ExpandGradFunctor>("ExpandGrad");
  m.add_functor<impl::ExpandDimsFunctor>("ExpandDims");
  m.add_functor<impl::ExpandDimsFunctor>("Unsqueeze");
  m.add_functor<impl::UnsqueezeMultipleFunctor>("UnsqueezeMultiple");
  m.add_functor<impl::SqueezeFunctor>("Squeeze");
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
  m.add_functor<impl::CopyFunctor>("Copy");
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
  m.add_functor<impl::TrilFunctor>("Tril");
  m.add_functor<impl::TriuFunctor>("Triu");
  m.add_functor<impl::DiagFunctor>("Diag");
  m.add_functor<impl::DiagGradFunctor>("DiagGrad");
  m.add_functor<impl::DiagonalFunctor>("Diagonal");
  m.add_functor<impl::DiagonalGradFunctor>("DiagonalGrad");
  m.add_functor<impl::TensorGetItemFunctor>("TensorGetItem");
  m.add_functor<impl::DimScatterFunctor>("DimScatter");
  m.add_functor<impl::DimScatterAddFunctor>("DimScatterAdd");
  m.add_functor<impl::DimScatterMulFunctor>("DimScatterMul");
  m.add_functor<impl::DimScatterUpdateScalarFunctor>("DimScatterUpdateScalar");
  m.add_functor<impl::DimScatterAddScalarFunctor>("DimScatterAddScalar");
  m.add_functor<impl::DimScatterAddLikeFunctor>("DimScatterAddLike");
  m.add_functor<impl::DimScatterMulScalarFunctor>("DimScatterMulScalar");
  m.add_functor<impl::TensorSetItemFunctor>("TensorSetItem");
  m.add_functor<impl::CastLikeFunctor>("CastLike");
  m.add_functor<impl::ElementwiseMinimumGradFunctor>("ElementwiseMinGrad");
  m.add_functor<impl::ElementwiseMaximumGradFunctor>("ElementwiseMaxGrad");
  m.add_functor<impl::BroadcastPowXGradFunctor>("BroadcastPowXGrad");
  m.add_functor<impl::BroadcastPowYGradFunctor>("BroadcastPowYGrad");
  m.add_functor<impl::DivGradFunctor>("DivGrad");
  m.add_functor<impl::IdentityFunctor>("Identity");
  m.add_functor<impl::AmpWhiteIdentityFunctor>("AmpWhiteIdentity");
  m.add_functor<impl::ReduceSumLikeFunctor>("ReduceSumLike");
  m.add_functor<impl::BroadcastReduceSumLikeFunctor>("BroadcastReduceSumLike");
  m.add_functor<impl::SplitFunctor>("Split");
  m.add_functor<impl::UnbindFunctor>("Unbind");
  m.add_functor<impl::ChunkFunctor>("Chunk");
  m.add_functor<impl::SplitLikeFunctor>("SplitLike");
  m.add_functor<impl::SplitWithSizeFunctor>("SplitWithSize");
  m.add_functor<impl::BatchGatherFunctor>("BatchGather");
  m.add_functor<impl::UnsortedBatchSegmentSumFunctor>("UnsortedBatchSegmentSum");
  m.add_functor<impl::MaskedFillFunctor>("MaskedFill");
  m.add_functor<impl::MeshgridFunctor>("Meshgrid");
  m.add_functor<impl::IndexSelectFunctor>("IndexSelect");
  m.add_functor<impl::ToFunctor, impl::To2Functor, impl::To3Functor, impl::To4Functor>("To");
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
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
