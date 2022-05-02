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
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/autograd/autograd_mode.h"

namespace oneflow {
namespace one {
namespace functional {

bool IsStaticZerosTensor(const std::shared_ptr<Tensor>& x) {
  return nullptr != std::dynamic_pointer_cast<StaticZerosTensor>(x);
}

bool IsInplaceValid(const std::shared_ptr<Tensor>& x) {
  return !autograd::GradMode::is_enabled() || !(x->is_leaf() && x->requires_grad());
}

Maybe<void> CheckAxis(std::vector<int32_t>& axis, const Shape& shape) {
  int32_t ndim = shape.NumAxes();
  if (axis.size() == 0) {
    for (int32_t i = 0; i < axis.size(); ++i) { axis[i] = i; }
  } else {
    for (int i = 0; i < axis.size(); ++i) {
      CHECK_OR_RETURN((-ndim < axis[i]) || (axis[i] < ndim - 1))
          << "Dimension out of range (expected to be in range of [" << -ndim << ", " << ndim - 1
          << "], but got " << axis[i];
      if (axis[i] < 0) { axis[i] += ndim; }
    }
  }
  return Maybe<void>::Ok();
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
      << "RuntimeError: result type " << x_cast->dtype()->name()
      << " can't be cast to the desired output type " << x->dtype()->name();
  return Maybe<void>::Ok();
}

bool IsShapeCanExpandTo(const Shape& shape, const Shape& expand_shape) {
  if (shape == expand_shape) { return true; }
  if (expand_shape.NumAxes() < shape.NumAxes()) { return false; }
  int shift = expand_shape.NumAxes() - shape.NumAxes();
  for (int i = expand_shape.NumAxes() - 1; i >= 0; --i) {
    int index = i - shift;
    if (index >= 0) {
      int dim_a = expand_shape.At(i);
      int dim_b = shape.At(index);
      if (dim_a != dim_b && (dim_a <= 0 || dim_b != 1)) { return false; }
    } else {
      if (expand_shape.At(i) <= 0) { return false; }
    }
  }
  return true;
}

Maybe<void> CheckShapeCanExpandTo(const Shape& shape, const Shape& expand_shape) {
  CHECK_OR_RETURN(IsShapeCanExpandTo(shape, expand_shape))
      << "Can not expand shape " << shape.ToString() << " to " << expand_shape.ToString();
  return Maybe<void>::Ok();
}

Optional<Stride> ComputeStride(const Shape& shape, const Stride& stride,
                               const Shape& target_shape) {
  /*************************************************
   * Description: in some case, view operate is not allowed, so need to check it's validation,
   * the check refers to torch(aten/src/ATen/native/TensorShape.cpp)
   *************************************************/
  if (stride.NumAxes() == 0) { return NullOpt; }
  int64_t elem_count = shape.elem_cnt();
  int64_t ndim = shape.NumAxes();
  int64_t tgt_ndim = target_shape.NumAxes();
  DimVector shape_vec = shape.dim_vec();
  DimVector tgt_shape_vec = target_shape.dim_vec();
  DimVector stride_vec = stride.StrideVec();
  if (elem_count == 0) { return NullOpt; }

  int64_t view_d = tgt_ndim - 1;
  int64_t chunk_base_stride = stride_vec.back();
  DimVector newstride(tgt_ndim);
  // stride for each subspace in the chunk
  // numel in current chunk
  int64_t tensor_numel = 1;
  int64_t view_numel = 1;
  for (int64_t tensor_d = ndim - 1; tensor_d >= 0; tensor_d--) {
    tensor_numel *= shape_vec[tensor_d];
    // if end of tensor size chunk, check view
    if ((tensor_d == 0)
        || (shape_vec[tensor_d - 1] != 1
            && stride_vec[tensor_d - 1] != tensor_numel * chunk_base_stride)) {
      while (view_d >= 0 && (view_numel < tensor_numel || tgt_shape_vec[view_d] == 1)) {
        newstride[view_d] = view_numel * chunk_base_stride;
        view_numel *= tgt_shape_vec[view_d];
        view_d--;
      }
      if (view_numel != tensor_numel) { return NullOpt; }
      if (tensor_d > 0) {
        chunk_base_stride = stride_vec[tensor_d - 1];
        tensor_numel = 1;
        view_numel = 1;
      }
    }
  }
  if (view_d != -1) { return NullOpt; }
  Stride target_stride(newstride);
  return target_stride;
}

Maybe<Shape> InferShape(const std::shared_ptr<one::Tensor>& x, const Shape& shape) {
  int need_infer_axis = -1;
  size_t count = 1;
  for (int i = 0; i < shape.NumAxes(); ++i) {
    if (shape.At(i) < -1) {
      return Error::RuntimeError() << "Invalid shape dimension " << shape.At(i);
    } else if (shape.At(i) == -1) {
      CHECK_EQ_OR_RETURN(need_infer_axis, -1)
          << Error::RuntimeError() << "only one dimension can be inferred";
      need_infer_axis = i;
    } else {
      count *= shape.At(i);
    }
  }
  size_t x_count = x->shape()->Count(0);
  Shape infered_shape = shape;
  if (need_infer_axis == -1) {
    CHECK_EQ_OR_RETURN(shape.Count(0), x_count)
        << "\n Shape " << shape.ToString() << " is invalid for input shape "
        << x->shape()->ToString();
  } else {
    infered_shape.Set(need_infer_axis, x_count / count);
    CHECK_EQ_OR_RETURN(infered_shape.Count(0), x_count)
        << "\n Shape " << shape.ToString() << " is invalid for input shape "
        << x->shape()->ToString();
  }
  return infered_shape;
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
