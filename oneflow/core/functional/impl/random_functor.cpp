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
#include "oneflow/core/common/global.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/random_generator.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/functional/impl/unary_functor.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/user/kernels/bernoulli_kernel.h"
#include "oneflow/user/kernels/distributions/normal_kernel.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class BernoulliFunctor {
 public:
  BernoulliFunctor() {
    bernoulli_op_ = CHECK_JUST(one::OpBuilder("bernoulli").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const DataType& dtype,
                           const Optional<one::Generator>& generator) const {
    MutableAttrMap bernoulli_attrs;
    JUST(bernoulli_attrs.SetAttr<DataType>("dtype", dtype));

    std::shared_ptr<one::Generator> gen;
    if (!generator) {
      gen = JUST(one::DefaultAutoGenerator());
    } else {
      gen = JUST(generator.value());
    }

    JUST(bernoulli_attrs.SetAttr<int64_t>("seed", gen->current_seed()));

    const auto& bernoulli_kernel_state = std::make_shared<BernoulliKernelState>(gen);

    return OpInterpUtil::Dispatch<Tensor>(
        *bernoulli_op_, {x}, OpExprInterpContext(bernoulli_attrs, bernoulli_kernel_state));
  }

 private:
  std::shared_ptr<OpExpr> bernoulli_op_;
};

class RandNFunctor {
 public:
  RandNFunctor() { op_ = CHECK_JUST(one::OpBuilder("normal").Output("out").Build()); }
  Maybe<Tensor> operator()(const Shape& shape, const Optional<DataType>& dtype,
                           const Optional<Symbol<Device>>& device,
                           const Optional<one::Generator>& generator) const {
    DataType dtype_val = DataType::kFloat;
    if (dtype.has_value()) {
      dtype_val = JUST(dtype.value());
      if (dtype_val != DataType::kFloat && dtype_val != DataType::kDouble) {
        OF_UNIMPLEMENTED() << dtype_val << "not supported in randn";
      }
    }

    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("mean", 0));
    JUST(attrs.SetAttr<double>("std", 1));
    JUST(attrs.SetAttr<Shape>("shape", shape));
    JUST(attrs.SetAttr<DataType>("dtype", dtype_val));

    std::shared_ptr<one::Generator> gen;

    if (!generator) {
      gen = JUST(one::DefaultAutoGenerator());
    } else {
      gen = JUST(generator.value());
    }

    JUST(attrs.SetAttr<int64_t>("seed", gen->current_seed()));

    const auto& normal_kernel_state = std::make_shared<NormalKernelState>(gen);

    if (device.has_value()) {
      Symbol<Device> device_symbol = JUST(device.value());
      return OpInterpUtil::Dispatch<Tensor>(
          *op_, {}, OpExprInterpContext(attrs, device_symbol, normal_kernel_state));
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*op_, {},
                                            OpExprInterpContext(attrs, normal_kernel_state));
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ConsistentRandNFunctor {
 public:
  ConsistentRandNFunctor() { op_ = CHECK_JUST(one::OpBuilder("normal").Output("out").Build()); }
  Maybe<Tensor> operator()(const Shape& shape, const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<cfg::SbpParallel>>& sbp_tuple,
                           const Optional<DataType>& dtype,
                           const Optional<one::Generator>& generator) const {
    DataType dtype_val = DataType::kFloat;
    if (dtype.has_value()) {
      dtype_val = JUST(dtype.value());
      if (dtype_val != DataType::kFloat && dtype_val != DataType::kDouble) {
        OF_UNIMPLEMENTED() << dtype_val << "not supported in randn";
      }
    }

    MutableAttrMap attrs;
    JUST(attrs.SetAttr<double>("mean", 0));
    JUST(attrs.SetAttr<double>("std", 1));
    JUST(attrs.SetAttr<Shape>("shape", shape));
    JUST(attrs.SetAttr<DataType>("dtype", dtype_val));

    std::shared_ptr<one::Generator> gen;
    if (!generator) {
      gen = JUST(one::DefaultAutoGenerator());
    } else {
      gen = JUST(generator.value());
    }

    JUST(attrs.SetAttr<int64_t>("seed", gen->current_seed()));

    const auto& normal_kernel_state = std::make_shared<NormalKernelState>(gen);

    const auto& nd_sbp = JUST(GetNdSbp(sbp_tuple));
    if (!JUST(*Global<Maybe<bool>, MultiClient>::Get())) {
      JUST(attrs.SetAttr<std::string>("nd_sbp", nd_sbp->DebugString()));
    }
    return OpInterpUtil::Dispatch<Tensor>(
        *op_, {}, OpExprInterpContext(attrs, placement, nd_sbp, normal_kernel_state));
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::BernoulliFunctor>("Bernoulli");
  m.add_functor<impl::RandNFunctor>("RandN");
  m.add_functor<impl::ConsistentRandNFunctor>("ConsistentRandN");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
