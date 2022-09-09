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
#include "fmt/core.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/autograd/autograd_mode.h"
#include "oneflow/core/common/wrap_dim_utils.h"

namespace oneflow {
namespace one {
namespace functional {

bool IsStaticZerosTensor(const std::shared_ptr<Tensor>& x) {
  return nullptr != std::dynamic_pointer_cast<StaticZerosTensor>(x);
}

bool IsInplaceValid(const std::shared_ptr<Tensor>& x) {
  return !autograd::GradMode::is_enabled() || !(x->is_leaf() && x->requires_grad());
}

bool IsScalarTensor(const std::shared_ptr<Tensor>& x) {
  return x->shape()->NumAxes() == 0 && x->shape()->elem_cnt() == 1;
}

Maybe<std::vector<int32_t>> CheckAxis(const std::vector<int32_t>& axis, const int32_t& ndim) {
  const int32_t naxis = axis.size();
  int32_t reduce_ndim = naxis;
  if (naxis == 0 || ndim == 0) { reduce_ndim = ndim; };
  std::vector<int32_t> reduce_axis(reduce_ndim);
  if (naxis == 0) {
    std::iota(reduce_axis.begin(), reduce_axis.end(), 0);
  } else {
    JUST(dim_list_to_bitset(axis, ndim));  // checking axis[dim]'s validation
    for (int32_t i = 0; i < naxis; i++) {
      if (i < reduce_ndim) { reduce_axis[i] = JUST(maybe_wrap_dim(axis[i], ndim)); };
    }
  }
  return reduce_axis;
}

Maybe<void> CheckInplaceValid(const std::shared_ptr<Tensor>& x) {
  CHECK_OR_RETURN(IsInplaceValid(x))
      << Error::RuntimeError()
      << "a leaf Tensor that requires grad is being used in an in-place operation";
  return Maybe<void>::Ok();
}

Maybe<void> CheckInplaceCastValid(const std::shared_ptr<Tensor>& x,
                                  const std::shared_ptr<Tensor>& x_cast) {
  CHECK_OR_RETURN(*x->dtype() == *x_cast->dtype())
      << Error::RuntimeError() << "result type " << x_cast->dtype()->name()
      << " can't be cast to the desired output type " << x->dtype()->name();
  return Maybe<void>::Ok();
}

Maybe<void> CheckInplaceShapeCanExpandTo(const Shape& shape, const Shape& expand_shape) {
  if (shape == expand_shape) { return Maybe<void>::Ok(); }

  CHECK_OR_RETURN(expand_shape.NumAxes() >= shape.NumAxes())
      << Error::RuntimeError() << "Can not expand origin shape " << shape.ToString() << " to "
      << expand_shape.ToString() << " in an inplace operation";

  int shift = expand_shape.NumAxes() - shape.NumAxes();
  for (int i = expand_shape.NumAxes() - 1; i >= 0; --i) {
    int index = i - shift;
    if (index >= 0) {
      int dim_a = expand_shape.At(i);
      int dim_b = shape.At(index);
      // NOTE(lixiang): When a dimension of tensor a and tensor b are not equal in size, dim_a needs
      // to be greater than 0, and dim_b should be equal to 1.
      CHECK_OR_RETURN(!(dim_a != dim_b && (dim_a <= 0 || dim_b != 1)))
          << Error::RuntimeError() << "Tensor with shape " << expand_shape.ToString()
          << " doesn't match the broadcast shape in an inplace operation";
    } else {
      CHECK_OR_RETURN(expand_shape.At(i) > 0);  // NOLINT(maybe-need-error-msg)
    }
  }

  return Maybe<void>::Ok();
}

Optional<Stride> ComputeStride(const Shape& shape, const Stride& stride,
                               const Shape& target_shape) {
  /*************************************************
   * Description: in some case, view operate is not allowed, so need to check it's validation,
   * the check refers to torch(aten/src/ATen/native/TensorShape.cpp)
   *************************************************/
  if (stride.size() == 0) {
    // for scalar input tensor
    return Stride(target_shape.NumAxes(), 1);
  }
  int64_t elem_count = shape.elem_cnt();
  int64_t ndim = shape.NumAxes();
  int64_t tgt_ndim = target_shape.NumAxes();
  DimVector shape_vec = shape.dim_vec();
  DimVector tgt_shape_vec = target_shape.dim_vec();
  if (elem_count == 0) { return NullOpt; }

  int64_t view_d = tgt_ndim - 1;
  int64_t chunk_base_stride = stride.back();
  Stride target_stride(tgt_ndim);
  // stride for each subspace in the chunk
  // numel in current chunk
  int64_t tensor_numel = 1;
  int64_t view_numel = 1;
  for (int64_t tensor_d = ndim - 1; tensor_d >= 0; tensor_d--) {
    tensor_numel *= shape_vec[tensor_d];
    // if end of tensor size chunk, check view
    if ((tensor_d == 0)
        || (shape_vec[tensor_d - 1] != 1
            && stride[tensor_d - 1] != tensor_numel * chunk_base_stride)) {
      while (view_d >= 0 && (view_numel < tensor_numel || tgt_shape_vec[view_d] == 1)) {
        target_stride[view_d] = view_numel * chunk_base_stride;
        view_numel *= tgt_shape_vec[view_d];
        view_d--;
      }
      if (view_numel != tensor_numel) { return NullOpt; }
      if (tensor_d > 0) {
        chunk_base_stride = stride[tensor_d - 1];
        tensor_numel = 1;
        view_numel = 1;
      }
    }
  }
  if (view_d != -1) { return NullOpt; }
  return target_stride;
}

Maybe<Shape> InferShapeUnspecifiedDim(const int64_t& elem_count, const Shape& shape) {
  int need_infer_axis = -1;
  int64_t target_elem_count = 1;
  for (int i = 0; i < shape.NumAxes(); ++i) {
    if (shape.At(i) < -1) {
      return Error::RuntimeError() << "Invalid shape dimension " << shape.At(i);
    } else if (shape.At(i) == -1) {
      CHECK_OR_RETURN_ERROR(need_infer_axis == -1)
          << Error::RuntimeError() << "only one dimension can be inferred";
      need_infer_axis = i;
    } else {
      target_elem_count *= shape.At(i);
    }
  }
  Shape infered_shape = shape;
  if (need_infer_axis == -1) {
    if (elem_count > 0) {
      // For 0-size tensor, we don't need to check the element size.
      CHECK_OR_RETURN_ERROR(target_elem_count == elem_count)
          << Error::RuntimeError() << "shape '" << shape.ToString()
          << "' is invalid for input of size " << elem_count;
    }
  } else {
    infered_shape.Set(need_infer_axis, elem_count / target_elem_count);
    CHECK_OR_RETURN_ERROR(target_elem_count * infered_shape.At(need_infer_axis) == elem_count)
        << Error::RuntimeError() << "shape '" << shape.ToString()
        << "' is invalid for input of size " << elem_count;
  }
  return infered_shape;
}

Maybe<std::tuple<Shape, bool, bool>> InferUnifiedShapeForBroadcasting(const Shape& input_shape,
                                                                      const Shape& other_shape) {
  if (input_shape == other_shape) { return std::make_tuple(input_shape, false, false); }

  const auto num_axes = std::make_pair(input_shape.NumAxes(), other_shape.NumAxes());

  if (num_axes.first < num_axes.second) {
    auto new_input_shape = Shape::Ones(num_axes.second);
    std::copy(input_shape.begin(), input_shape.end(),
              new_input_shape.begin() + (num_axes.second - num_axes.first));
    return InferUnifiedShapeForBroadcasting(new_input_shape, other_shape);
  }

  if (num_axes.first > num_axes.second) {
    auto new_other_shape = Shape::Ones(num_axes.first);
    std::copy(other_shape.begin(), other_shape.end(),
              new_other_shape.begin() + (num_axes.first - num_axes.second));
    return InferUnifiedShapeForBroadcasting(input_shape, new_other_shape);
  }

  // num_axes.first == num_axes.second
  Shape target;
  auto need_to_broadcast = std::make_pair(false, false);

  for (size_t i = 0; i < num_axes.first; ++i) {
    const auto num_in_curr_dim = std::make_pair(input_shape.At(i), other_shape.At(i));

    if (num_in_curr_dim.first == num_in_curr_dim.second) {
      target.push_back(num_in_curr_dim.first);
      continue;
    }

    if (num_in_curr_dim.first != 1 && num_in_curr_dim.second != 1) {
      return Error::RuntimeError()
             << fmt::format("input and other can't be broadcasted to a single shape. [input's "
                            "shape: {}, other's shape: {}].",
                            input_shape.ToString(), other_shape.ToString());
    }

    need_to_broadcast.first = num_in_curr_dim.first == 1 ? true : need_to_broadcast.first;
    need_to_broadcast.second = num_in_curr_dim.second == 1 ? true : need_to_broadcast.second;
    target.push_back(
        num_in_curr_dim.first == 1
            ? num_in_curr_dim.second
            : num_in_curr_dim.first);  // num_in_curr_dim.first and num_in_curr_dim.second can't
                                       // be 1 at the same time
  }
  return std::make_tuple(target, need_to_broadcast.first, need_to_broadcast.second);
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
