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

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/framework/op_expr.h"

namespace oneflow {
namespace op_expr_helper {

Maybe<one::UserOpExpr> AddNOp(int32_t n);
Maybe<one::UserOpExpr> AddNOp(int32_t n, const std::string& name);

Maybe<one::UserOpExpr> AddOp();
Maybe<one::UserOpExpr> AddOp(const std::string& name);

Maybe<one::UserOpExpr> ZerosOp(const Shape& shape, const DataType& dtype);
Maybe<one::UserOpExpr> ZerosOp(const Shape& shape, const DataType& dtype, const std::string& name);

Maybe<one::UserOpExpr> ZeroLikeOp();
Maybe<one::UserOpExpr> ZeroLikeOp(const std::string& name);

Maybe<one::UserOpExpr> EmptyOp(const Shape& shape, const DataType& dtype);
Maybe<one::UserOpExpr> EmptyOp(const Shape& shape, const DataType& dtype, const std::string& name);

Maybe<one::UserOpExpr> OnesLikeOp();
Maybe<one::UserOpExpr> OnesLikeOp(const std::string& name);

template<typename T>
Maybe<one::UserOpExpr> ConstantOp(const Shape& shape, const T& value);
template<typename T>
Maybe<one::UserOpExpr> ConstantOp(const Shape& shape, const T& value, const std::string& name);

Maybe<one::UserOpExpr> OnesOp(const Shape& shape, const DataType& dtype);
Maybe<one::UserOpExpr> OnesOp(const Shape& shape, const DataType& dtype, const std::string& name);

Maybe<one::UserOpExpr> IdentityOp();
Maybe<one::UserOpExpr> IdentityOp(const std::string& name);

Maybe<one::UserOpExpr> ReshapeOp(const Shape& shape);
Maybe<one::UserOpExpr> ReshapeOp(const Shape& shape, const std::string& name);

Maybe<one::UserOpExpr> ReshapeLikeOp();
Maybe<one::UserOpExpr> ReshapeLikeOp(const std::string& name);

Maybe<one::UserOpExpr> ReduceSumOp(const std::vector<int32_t>& reduce_axes, const bool& keepdims);
Maybe<one::UserOpExpr> ReduceSumOp(const std::vector<int32_t>& reduce_axes, const bool& keepdims,
                                   const std::string& name);

Maybe<one::UserOpExpr> ReduceSumLikeOp(const std::vector<int32_t>& axis);
Maybe<one::UserOpExpr> ReduceSumLikeOp(const std::vector<int32_t>& axis, const std::string& name);

Maybe<one::UserOpExpr> ScalarPowOp(const double& exponent);
Maybe<one::UserOpExpr> ScalarPowOp(const double& exponent, const std::string& name);

template<typename T>
Maybe<one::UserOpExpr> ScalarAddOp(const T& scalar);

template<typename T>
Maybe<one::UserOpExpr> ScalarAddOp(const T& scalar, const std::string& name);

template<typename T>
Maybe<one::UserOpExpr> ScalarMulOp(const T& scalar);

template<typename T>
Maybe<one::UserOpExpr> ScalarMulOp(const T& scalar, const std::string& name);

Maybe<one::UserOpExpr> RsqrtOp();
Maybe<one::UserOpExpr> RsqrtOp(const std::string& name);

Maybe<one::UserOpExpr> BroadcastAddOp();
Maybe<one::UserOpExpr> BroadcastAddOp(const std::string& name);

Maybe<one::UserOpExpr> BroadcastSubOp();
Maybe<one::UserOpExpr> BroadcastSubOp(const std::string& name);

Maybe<one::UserOpExpr> BroadcastMulOp();
Maybe<one::UserOpExpr> BroadcastMulOp(const std::string& name);

Maybe<one::UserOpExpr> BroadcastDivOp();
Maybe<one::UserOpExpr> BroadcastDivOp(const std::string& name);

Maybe<one::UserOpExpr> BroadcastLikeOp(const std::vector<int32_t>& axis);
Maybe<one::UserOpExpr> BroadcastLikeOp(const std::vector<int32_t>& axis, const std::string& name);

Maybe<one::UserOpExpr> BroadcastEqualOp();
Maybe<one::UserOpExpr> BroadcastEqualOp(const std::string& name);

Maybe<one::UserOpExpr> CastOp(const DataType& to_type);
Maybe<one::UserOpExpr> CastOp(const DataType& to_type, const std::string& name);

Maybe<one::UserOpExpr> CastLikeOp();
Maybe<one::UserOpExpr> CastLikeOp(const std::string& name);

Maybe<one::UserOpExpr> CopyOp(const std::string& device_type, const int64_t device_id);
Maybe<one::UserOpExpr> CopyOp(const std::string& device_type, const int64_t device_id,
                              const std::string& name);

Maybe<one::UserOpExpr> NormalizationGradOp(const int32_t& axis, const float& epsilon);
Maybe<one::UserOpExpr> NormalizationGradOp(const int32_t& axis, const float& epsilon,
                                           const std::string& name);

Maybe<one::UserOpExpr> BroadcastDivGradOp();
Maybe<one::UserOpExpr> BroadcastDivGradOp(const std::string& name);

Maybe<one::UserOpExpr> ConcatOp(const int& n, const int64_t& axis, const int64_t& max_dim_size);
Maybe<one::UserOpExpr> ConcatOp(const int& n, const int64_t& axis, const int64_t& max_dim_size,
                                const std::string& name);

Maybe<one::UserOpExpr> ScalarAddByTensorOp();
Maybe<one::UserOpExpr> ScalarAddByTensorOp(const std::string& name);

Maybe<one::UserOpExpr> ScalarSubByTensorOp();
Maybe<one::UserOpExpr> ScalarSubByTensorOp(const std::string& name);

Maybe<one::UserOpExpr> ScalarMulByTensorOp();
Maybe<one::UserOpExpr> ScalarMulByTensorOp(const std::string& name);

Maybe<one::UserOpExpr> ScalarDivByTensorOp();
Maybe<one::UserOpExpr> ScalarDivByTensorOp(const std::string& name);

Maybe<one::UserOpExpr> MultiplyOp();
Maybe<one::UserOpExpr> MultiplyOp(const std::string& name);

Maybe<one::UserOpExpr> ConvNdOp(const int& filters, const std::vector<int32_t>& kernel_size,
                                const std::vector<int32_t>& strides,
                                const std::vector<int32_t>& padding_before,
                                const std::vector<int32_t>& dilation_rate, const int& groups,
                                const std::string& data_format);
Maybe<one::UserOpExpr> ConvNdOp(const int& filters, const std::vector<int32_t>& kernel_size,
                                const std::vector<int32_t>& strides,
                                const std::vector<int32_t>& padding_before,
                                const std::vector<int32_t>& dilation_rate, const int& groups,
                                const std::string& data_format, const std::string& name);

Maybe<one::UserOpExpr> ConvNdFilterGradOp(const std::vector<int32_t>& kernel_size,
                                          const std::vector<int32_t>& strides,
                                          const std::vector<int32_t>& padding_before,
                                          const std::vector<int32_t>& dilation_rate,
                                          const int& groups, const std::string& data_format);
Maybe<one::UserOpExpr> ConvNdFilterGradOp(const std::vector<int32_t>& kernel_size,
                                          const std::vector<int32_t>& strides,
                                          const std::vector<int32_t>& padding_before,
                                          const std::vector<int32_t>& dilation_rate,
                                          const int& groups, const std::string& data_format,
                                          const std::string& name);

Maybe<one::UserOpExpr> ConvNdDataGradOp(const std::vector<int32_t>& kernel_size,
                                        const std::vector<int32_t>& strides,
                                        const std::vector<int32_t>& padding_before,
                                        const std::vector<int32_t>& dilation_rate,
                                        const int& groups, const std::string& data_format);
Maybe<one::UserOpExpr> ConvNdDataGradOp(const std::vector<int32_t>& kernel_size,
                                        const std::vector<int32_t>& strides,
                                        const std::vector<int32_t>& padding_before,
                                        const std::vector<int32_t>& dilation_rate,
                                        const int& groups, const std::string& data_format,
                                        const std::string& name);

Maybe<one::UserOpExpr> CTCLossGradOp(const int32_t& blank, const bool& zero_infinity);
Maybe<one::UserOpExpr> CTCLossGradOp(const int32_t& blank, const bool& zero_infinity,
                                     const std::string& name);

Maybe<one::UserOpExpr> SparseSoftmaxCrossEntropyGradOp(const int64_t& depth);
Maybe<one::UserOpExpr> SparseSoftmaxCrossEntropyGradOp(const int64_t& depth,
                                                       const std::string& name);
Maybe<one::UserOpExpr> SparseSoftmaxCrossEntropyMsGradOp(const int64_t& depth);
Maybe<one::UserOpExpr> SparseSoftmaxCrossEntropyMsGradOp(const int64_t& depth,
                                                         const std::string& name);
Maybe<one::UserOpExpr> PReLUGradOp();
Maybe<one::UserOpExpr> PReLUGradOp(const std::string& name);

Maybe<one::UserOpExpr> UpsampleGradOp(const float& height_scale, const float& width_scale,
                                      const bool& align_corners, const std::string& data_format,
                                      const std::string& interpolation);
Maybe<one::UserOpExpr> UpsampleGradOp(const float& height_scale, const float& width_scale,
                                      const bool& align_corners, const std::string& data_format,
                                      const std::string& interpolation, const std::string& name);

Maybe<one::UserOpExpr> DimScatterAddLikeOp(const int32_t dim);
Maybe<one::UserOpExpr> DimScatterAddLikeOp(const int32_t dim, const std::string& name);
Maybe<one::UserOpExpr> TransposeOp(const std::vector<int32_t>& perm);
Maybe<one::UserOpExpr> TransposeOp(const std::vector<int32_t>& perm, const std::string& name);

Maybe<one::UserOpExpr> SplitLikeOp(const int n, const int64_t axis);
Maybe<one::UserOpExpr> SplitLikeOp(const int n, const int64_t axis, const std::string& name);

Maybe<one::UserOpExpr> WhereOp();
Maybe<one::UserOpExpr> WhereOp(const std::string& name);

Maybe<one::UserOpExpr> ExpandGradOp(const std::vector<int32_t>& out_shape,
                                    const std::vector<int32_t>& stride);
Maybe<one::UserOpExpr> ExpandGradOp(const std::vector<int32_t>& out_shape,
                                    const std::vector<int32_t>& stride, const std::string& name);

Maybe<one::UserOpExpr> UnaryGradOp(const std::string& unary_op_type);
Maybe<one::UserOpExpr> UnaryGradOp(const std::string& unary_op_type, const std::string& name);

Maybe<one::UserOpExpr> BinaryXGradOp(const std::string& binary_op_type);
Maybe<one::UserOpExpr> BinaryXGradOp(const std::string& binary_op_type, const std::string& name);

Maybe<one::UserOpExpr> BinaryYGradOp(const std::string& binary_op_type);
Maybe<one::UserOpExpr> BinaryYGradOp(const std::string& binary_op_type, const std::string& name);

Maybe<one::UserOpExpr> MatmulOp(const bool& transpose_a, const bool& transpose_b,
                                const double& alpha);
Maybe<one::UserOpExpr> MatmulOp(const bool& transpose_a, const bool& transpose_b,
                                const double& alpha, const std::string& name);

Maybe<one::UserOpExpr> BatchMatmulOp(const bool& transpose_a, const bool& transpose_b,
                                     const double& alpha);
Maybe<one::UserOpExpr> BatchMatmulOp(const bool& transpose_a, const bool& transpose_b,
                                     const double& alpha, const std::string& name);

Maybe<one::UserOpExpr> BroadcastMatmulOp(const bool& transpose_a, const bool& transpose_b,
                                         const double& alpha);
Maybe<one::UserOpExpr> BroadcastMatmulOp(const bool& transpose_a, const bool& transpose_b,
                                         const double& alpha, const std::string& name);

Maybe<one::UserOpExpr> BroadcastMatmulGradBOp(const double& alpha);
Maybe<one::UserOpExpr> BroadcastMatmulGradBOp(const double& alpha, const std::string& name);

Maybe<one::UserOpExpr> PoolNdGradOp(const std::string& mode, const std::string& data_format,
                                    const std::string& padding,
                                    const std::vector<int32_t>& padding_before,
                                    const std::vector<int32_t>& padding_after,
                                    const std::vector<int32_t>& pool_size,
                                    const std::vector<int32_t>& strides, const bool& ceil_mode);
Maybe<one::UserOpExpr> PoolNdGradOp(const std::string& mode, const std::string& data_format,
                                    const std::string& padding,
                                    const std::vector<int32_t>& padding_before,
                                    const std::vector<int32_t>& padding_after,
                                    const std::vector<int32_t>& pool_size,
                                    const std::vector<int32_t>& strides, const bool& ceil_mode,
                                    const std::string& name);

Maybe<one::UserOpExpr> UnsortedSegmentSumLikeOp(const int64_t& axis);
Maybe<one::UserOpExpr> UnsortedSegmentSumLikeOp(const int64_t& axis, const std::string& name);

Maybe<one::UserOpExpr> SoftmaxGradOp();
Maybe<one::UserOpExpr> SoftmaxGradOp(const std::string& name);
}  // namespace op_expr_helper
}  // namespace oneflow
