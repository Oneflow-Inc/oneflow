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
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/autograd/autograd_mode.h"
#include "oneflow/core/common/wrap_dim_utils.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/ccl/ccl.h"
#include "oneflow/core/job/rank_group.h"

namespace oneflow {
namespace one {
namespace functional {
namespace {

Maybe<Shape> InferUnifiedShapeForBroadcasting(const Shape& input_shape, const Shape& other_shape) {
  // same shapes need no broadcasting
  if (input_shape == other_shape) { return input_shape; }

  const auto unify_shapes_with_same_num_axes = [](const Shape& input_shape,
                                                  const Shape& other_shape) -> Maybe<Shape> {
    // num_axes.first == num_axes.second
    Shape target;
    for (size_t i = 0; i < input_shape.NumAxes() /* both input_shape and other_shape are ok */;
         ++i) {
      const auto num_in_curr_dim = std::make_pair(input_shape.At(i), other_shape.At(i));

      // A = (2, ), B = (2, ), A[0] == B[0], so C = (2, )
      if (num_in_curr_dim.first == num_in_curr_dim.second) {
        target.push_back(num_in_curr_dim.first);
        continue;
      }

      // A = (2, ), B = (3, ), A[0] != B[0] and A[0] != 1 and B[0] != 1, so raise RuntimeError
      if (num_in_curr_dim.first != 1 && num_in_curr_dim.second != 1) {
        return Error::RuntimeError()
               << fmt::format("input and other can't be broadcasted to a single shape. [input's "
                              "shape: {}, other's shape: {}].",
                              input_shape.ToString(), other_shape.ToString());
      }

      // A = (2, ), B = (1, ), A[0] != B[0] but B[0] == 1, so C = (2, )
      target.push_back(
          num_in_curr_dim.first == 1
              ? num_in_curr_dim.second
              : num_in_curr_dim.first);  // num_in_curr_dim.first and num_in_curr_dim.second can't
                                         // be 1 at the same time
    }
    return target;
  };

  const int64_t input_num_axes = input_shape.NumAxes();
  const int64_t other_num_axes = other_shape.NumAxes();

  if (input_num_axes == other_num_axes) {
    return unify_shapes_with_same_num_axes(input_shape, other_shape);
  }

  const int64_t unified_num_axes = std::max(input_num_axes, other_num_axes);

  // shape = (3, 4) and unified_num_axes = 3 ==> shape will be (1, 3, 4)
  const auto expand_shape_if_necessary = [unified_num_axes](const Shape& shape_to_expand) {
    const int64_t shape_to_expand_num_axes = shape_to_expand.NumAxes();
    if (shape_to_expand_num_axes < unified_num_axes) {
      auto new_shape = Shape::Ones(unified_num_axes);
      std::copy(shape_to_expand.begin(), shape_to_expand.end(),
                new_shape.begin() + (unified_num_axes - shape_to_expand_num_axes));
      return new_shape;
    }
    return shape_to_expand;
  };

  return unify_shapes_with_same_num_axes(expand_shape_if_necessary(input_shape),
                                         expand_shape_if_necessary(other_shape));
}

}  // namespace

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
      // For 0-size tensor, expand_shape.At(i) can equal to 0.
      CHECK_OR_RETURN(expand_shape.At(i) >= 0);  // NOLINT(maybe-need-error-msg)
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

Maybe<Shape> InferUnifiedShapeForBroadcasting(const std::vector<Shape>& shapes) {
  if (shapes.empty()) { return Error::RuntimeError() << "shapes should not be empty."; }
  if (shapes.size() == 1) { return JUST(VectorAt(shapes, 0)); }

  auto result =
      *JUST(InferUnifiedShapeForBroadcasting(JUST(VectorAt(shapes, 0)), JUST(VectorAt(shapes, 1))));

  // (1, 2) vs (3, 2) => (3, 2)
  if (shapes.size() == 2) { return result; }

  /*
    (1, 3) vs (3, 1) vs (3, 1, 1)

    1. (1, 3) vs (3, 1) => (3, 3)
    2. (3, 3) vs (3, 1, 1) => (3, 3, 3)
    3. (3, 3, 3) is the final result
  */
  for (auto iter = shapes.begin() + 2; iter != shapes.end(); ++iter) {
    result = *JUST(InferUnifiedShapeForBroadcasting(result, *iter));
  }
  return result;
}

/*
  if input shapes are [(1, 3), (3, 1), (3, 1, 1)]
  will return ((3, 3, 3), [true, true, true])
  means the shape to broadcast to is (3, 3, 3) and all three shapes need broadcasting
*/
Maybe<std::tuple<Shape, std::deque<bool>>> InferUnifiedShapeForBroadcastingWithInfo(
    const std::vector<Shape>& shapes) {
  const auto unified_shape = *JUST(InferUnifiedShapeForBroadcasting(shapes));
  std::deque<bool> need_to_broadcast;
  for (const auto& x : shapes) { need_to_broadcast.emplace_back(x != unified_shape); }
  return std::make_tuple(unified_shape, need_to_broadcast);
}

Maybe<void> BroadcastSeedToAllRanks(uint64_t* seed, int64_t root) {
  CHECK_NOTNULL_OR_RETURN(seed) << "seed is not allowed to be nullptr";
  const auto& rank_group = JUST(RankGroup::DefaultRankGroup());
  const auto& parallel_desc = JUST(RankGroup::GetDefaultParallelDesc(DeviceType::kCPU, rank_group));
  const auto& meta_transport_token =
      JUST(TransportToken::NewTransportToken(kTransportTokenTypeMeta));
  JUST(ccl::CpuBroadcast(seed, seed, sizeof(*seed), root, parallel_desc, meta_transport_token));
  return Maybe<void>::Ok();
}

Maybe<std::vector<int32_t>> GetPermWhenTransposeAxisToLastDim(const int32_t& ndim,
                                                              const int32_t& axis) {
  auto wrap_dim = JUST(maybe_wrap_dim(axis, ndim));
  std::vector<int32_t> perm(ndim);
  for (int i = 0; i < ndim - 1; i++) {
    if (i < wrap_dim) {
      perm[i] = i;
    } else {
      perm[i] = i + 1;
    }
  }
  perm[ndim - 1] = wrap_dim;
  return perm;
}

Maybe<std::vector<int32_t>> GetInversedPerm(const std::vector<int32_t>& perm) {
  std::vector<int32_t> inversed_perm(perm.size());
  for (int i = 0; i < perm.size(); i++) { inversed_perm[perm[i]] = i; }
  return inversed_perm;
}

Maybe<std::tuple<std::shared_ptr<Tensor>, bool>> batchify(const std::shared_ptr<Tensor>& input,
                                                          const int64_t num_spatial_dims,
                                                          const std::string& func_name) {
  const int64_t dim_count_no_batch = num_spatial_dims + 1;
  const int64_t dim_count_batch = dim_count_no_batch + 1;
  const bool is_batched = (input->ndim() == dim_count_batch);
  CHECK_EQ_OR_RETURN(input->ndim() == dim_count_no_batch || is_batched, true) << fmt::format(
      "Expected `{}`D (unbatched) or `{}`D (batched) input to `{}`, but got input of size: `{}`",
      dim_count_no_batch, dim_count_batch, func_name, input->shape()->DebugStr());
  return std::make_tuple(is_batched ? input : JUST(functional::Unsqueeze(input, 0)), is_batched);
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
