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

#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/functional/impl/unary_functor.h"
#include "oneflow/core/functional/scalar.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class ConvBiasGradFunctor {
 public:
  ConvBiasGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("conv_bias_grad").Input("dy").Output("bias_diff").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy, const int32_t& num_spatial_dims,
                           const std::string& data_format) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("num_spatial_dims", num_spatial_dims));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ConvFilterGradFunctor {
 public:
  ConvFilterGradFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("conv_filter_grad").Input("dy").Input("x").Output("filter_diff").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x, const int32_t& num_spatial_dims,
                           const std::vector<int32_t>& kernel_size,
                           const std::vector<int32_t>& strides,
                           const std::vector<int32_t>& padding_before,
                           const std::vector<int32_t>& dilation_rate, const int32_t& groups,
                           const std::string& data_format) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("num_spatial_dims", num_spatial_dims));
    JUST(attrs.SetAttr<std::vector<int32_t>>("kernel_size", kernel_size));
    JUST(attrs.SetAttr<std::vector<int32_t>>("strides", strides));
    JUST(attrs.SetAttr<std::vector<int32_t>>("padding_before", padding_before));
    JUST(attrs.SetAttr<std::vector<int32_t>>("dilation_rate", dilation_rate));
    JUST(attrs.SetAttr<int32_t>("groups", groups));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ConvDataGradFunctor {
 public:
  ConvDataGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("conv_data_grad")
                         .Input("dy")
                         .Input("filter")
                         .Input("x_like")
                         .Output("dx")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& weight,
                           const std::shared_ptr<one::Tensor>& x, const int32_t& num_spatial_dims,
                           const std::vector<int32_t>& kernel_size,
                           const std::vector<int32_t>& strides,
                           const std::vector<int32_t>& padding_before,
                           const std::vector<int32_t>& dilation_rate, const int32_t& groups,
                           const std::string& data_format) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("num_spatial_dims", num_spatial_dims));
    JUST(attrs.SetAttr<std::vector<int32_t>>("kernel_size", kernel_size));
    JUST(attrs.SetAttr<std::vector<int32_t>>("strides", strides));
    JUST(attrs.SetAttr<std::vector<int32_t>>("padding_before", padding_before));
    JUST(attrs.SetAttr<std::vector<int32_t>>("dilation_rate", dilation_rate));
    JUST(attrs.SetAttr<int32_t>("groups", groups));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, weight, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class PoolNdGradFunctor {
 public:
  PoolNdGradFunctor() {
    for (const auto& mode : {"max", "avg"}) {
      for (const auto& ndims : {1, 2, 3}) {
        const auto& op_type_name = GetOpTypeName(mode, ndims);
        op_expr_map_[op_type_name] = CHECK_JUST(one::OpBuilder(op_type_name).Input("x").Input("y").Input("dy").Output("dx").Build());
      }
    }
  }
  static std::string GetOpTypeName(const std::string& mode, const int32_t& ndims) {
    return mode + "_pool_" + std::to_string(ndims) + "d_grad";
  }
  virtual ~PoolNdGradFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y,
                           const std::shared_ptr<one::Tensor>& dy,
                           const std::string& mode,
                           const int32_t& ndims,
                           const std::string& data_format,
                           const std::string& padding,
                           const std::vector<int32_t>& padding_before,
                           const std::vector<int32_t>& padding_after,
                           const std::vector<int32_t>& pool_size,
                           const std::vector<int32_t>& strides,  const bool& ceil_mode) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    JUST(attrs.SetAttr<std::string>("padding", padding));
    JUST(attrs.SetAttr<std::vector<int32_t>>("padding_before", padding_before));
    JUST(attrs.SetAttr<std::vector<int32_t>>("padding_after", padding_after));
    JUST(attrs.SetAttr<std::vector<int32_t>>("pool_size", pool_size));
    JUST(attrs.SetAttr<std::vector<int32_t>>("strides", strides));
    JUST(attrs.SetAttr<bool>("ceil_mode", ceil_mode));
    const auto& op_type_name = GetOpTypeName(mode, ndims);
    const auto& it = op_expr_map_.find(op_type_name);
    CHECK_OR_RETURN(it != op_expr_map_.end()) << op_type_name << " not found";
    const auto op = it->second;
    CHECK_NOTNULL_OR_RETURN(op);
    return OpInterpUtil::Dispatch<Tensor>(*op, {x, y, dy}, attrs);
  }

 protected:
  std::unordered_map<std::string, std::shared_ptr<OpExpr>> op_expr_map_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::ConvBiasGradFunctor>("ConvBiasGrad");
  m.add_functor<impl::ConvFilterGradFunctor>("ConvFilterGrad");
  m.add_functor<impl::ConvDataGradFunctor>("ConvDataGrad");
  m.add_functor<impl::PoolNdGradFunctor>("PoolNdGrad");

};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
