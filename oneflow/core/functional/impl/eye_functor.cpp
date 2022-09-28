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

#include "oneflow/core/common/just.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/common/throw.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/mutable_attr_map.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/job/sbp_parallel.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class EyeDevcieFunctor {
 public:
  EyeDevcieFunctor() { op_ = CHECK_JUST(one::OpBuilder("eye").Output("out").Build()); }
  Maybe<Tensor> operator()(const Scalar& rows, const Optional<Scalar>& cols,
                           const Symbol<DType>& dtype, const Optional<Symbol<Device>>& device,
                           const bool& requires_grad) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("rows", "cols", "dtype");
    attrs.SetAllAttrs(rows.As<int64_t>(), cols.value_or(rows).As<int64_t>(), dtype->data_type());
    OpExprInterpContext ctx(attrs);
    ctx.device = device;
    auto res = JUST(OpInterpUtil::Dispatch<Tensor>(*op_, {}, ctx));
    JUST(res->set_requires_grad(requires_grad));
    return res;
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class EyeDeviceStrFunctor {
 public:
  Maybe<Tensor> operator()(const Scalar& rows, const Optional<Scalar>& cols,
                           const Symbol<DType>& dtype, const std::string& device,
                           const bool& requires_grad) const {
    const Symbol<Device>& dev = JUST(Device::ParseAndNew(device));
    return JUST(functional::Eye(rows, cols, dtype, dev, requires_grad));
  }
};

class GlobalEyeSbpListFunctor {
 public:
  GlobalEyeSbpListFunctor() { op_ = CHECK_JUST(one::OpBuilder("eye").Output("out").Build()); }
  Maybe<Tensor> operator()(const Scalar& rows, const Optional<Scalar>& cols,
                           const Symbol<DType>& dtype, const bool& requires_grad,
                           const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp_tuple) const {
    CHECK_EQ_OR_RETURN(sbp_tuple.size(), placement->hierarchy()->NumAxes())
        << "len(sbp) == len(placement.hierarchy) required, but "
        << "len(sbp)==" << sbp_tuple.size() << ", "
        << "len(placement.hierarchy)==" << placement->hierarchy()->NumAxes();

    FOR_RANGE(int32_t, i, 0, sbp_tuple.size()) {
      CHECK_OR_RETURN(sbp_tuple.at(i)->has_broadcast_parallel())
          << "sbp of eye should be broadcast only";
    }
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("rows", "cols", "dtype", "nd_sbp");
    if (LazyMode::is_enabled()) {
      std::vector<std::string> nd_sbp(sbp_tuple.size());
      {
        for (int i = 0; i < sbp_tuple.size(); ++i) {
          nd_sbp.at(i) = SbpParallelToString(*sbp_tuple.at(i));
        }
      }
      attrs.SetAllAttrs(rows.As<int64_t>(), cols.value_or(rows).As<int64_t>(), dtype->data_type(),
                        nd_sbp);
    } else {
      attrs.SetAllAttrs(rows.As<int64_t>(), cols.value_or(rows).As<int64_t>(), dtype->data_type(),
                        NullOpt);
    }

    const auto& nd_sbp = JUST(GetNdSbp(sbp_tuple));
    auto res = JUST(
        OpInterpUtil::Dispatch<Tensor>(*op_, {}, OpExprInterpContext(attrs, placement, nd_sbp)));
    JUST(res->set_requires_grad(requires_grad));
    return res;
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class GlobalEyeSbpFunctor {
 public:
  Maybe<Tensor> operator()(const Scalar& rows, const Optional<Scalar>& cols,
                           const Symbol<DType>& dtype, const bool& requires_grad,
                           const Symbol<ParallelDesc>& placement,
                           const Symbol<SbpParallel>& sbp) const {
    std::vector<Symbol<SbpParallel>> sbp_tuple{sbp};
    return JUST(functional::Eye(rows, cols, dtype, requires_grad, placement, sbp_tuple));
  }
};

}  // namespace impl

class EyeInplaceFunctor {
 public:
  EyeInplaceFunctor() { op_ = CHECK_JUST(one::OpBuilder("eye").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x) const {
    JUST(CheckInplaceValid(x));
    std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
    outputs->at(0) = x;
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("rows", "cols", "dtype");
    attrs.SetAllAttrs(x->shape()->At(0), x->shape()->At(1), x->dtype()->data_type());
    OpExprInterpContext ctx(attrs);
    ctx.device = JUST(x->device());
    JUST(OpInterpUtil::Dispatch(*op_, {}, outputs.get(), ctx));
    return outputs->at(0);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

using namespace impl;

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<EyeDevcieFunctor, EyeDeviceStrFunctor, GlobalEyeSbpListFunctor,
                GlobalEyeSbpFunctor>("Eye");
  m.add_functor<EyeInplaceFunctor>("EyeInplace");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
