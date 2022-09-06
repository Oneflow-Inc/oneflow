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
#include <memory>
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/mutable_attr_map.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/random_generator.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/functional/impl/unary_functor.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/user/kernels/distributions/common.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class BernoulliFunctor {
 public:
  BernoulliFunctor() {
    bernoulli_op_ = CHECK_JUST(one::OpBuilder("bernoulli").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Symbol<DType>& dtype,
                           const Optional<one::Generator>& generator) const {
    const auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    auto& bernoulli_attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("dtype", "seed", "p");

    // p == -1 means bernoulli op doesn't use p to generate random number
    bernoulli_attrs.SetAllAttrs(dtype->data_type(), static_cast<int64_t>(gen->current_seed()),
                                static_cast<double>(-1));
    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);
    return OpInterpUtil::Dispatch<Tensor>(*bernoulli_op_, {x},
                                          OpExprInterpContext(bernoulli_attrs, distribution_state));
  }

 private:
  std::shared_ptr<OpExpr> bernoulli_op_;
};

class BernoulliProbFunctor {
 public:
  BernoulliProbFunctor() {
    bernoulli_op_ = CHECK_JUST(one::OpBuilder("bernoulli").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const double& p,
                           const Symbol<DType>& dtype,
                           const Optional<one::Generator>& generator) const {
    const auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);
    CHECK_OR_THROW(p >= 0.0 && p <= 1.0) << "bernoulli expects p to be in [0, 1], but got p=" << p;

    auto& bernoulli_attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("dtype", "seed", "p");
    bernoulli_attrs.SetAllAttrs(dtype->data_type(), static_cast<int64_t>(gen->current_seed()), p);
    return OpInterpUtil::Dispatch<Tensor>(*bernoulli_op_, {x},
                                          OpExprInterpContext(bernoulli_attrs, distribution_state));
  }

 private:
  std::shared_ptr<OpExpr> bernoulli_op_;
};

class RandFunctor {
 public:
  RandFunctor() { op_ = CHECK_JUST(one::OpBuilder("uniform").Output("out").Build()); }
  Maybe<Tensor> operator()(const Shape& shape, const Optional<Symbol<DType>>& dtype,
                           const Optional<Symbol<Device>>& device,
                           const Optional<one::Generator>& generator,
                           const bool& requires_grad) const {
    DataType dtype_val = DataType::kFloat;
    if (dtype.has_value()) {
      dtype_val = JUST(dtype)->data_type();
      if (dtype_val != DataType::kFloat && dtype_val != DataType::kDouble) {
        OF_UNIMPLEMENTED() << "Only support float and double in rand().";
      }
    }

    const auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("from", "to", "shape", "dtype", "seed");
    attrs.SetAllAttrs(static_cast<double>(0), static_cast<double>(1), shape, dtype_val,
                      static_cast<int64_t>(gen->current_seed()));

    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);

    OpExprInterpContext ctx(attrs, distribution_state);
    ctx.device = device;
    auto result = JUST(OpInterpUtil::Dispatch<Tensor>(*op_, {}, ctx));
    JUST(result->set_requires_grad(requires_grad));
    return result;
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class GlobalRandFunctor {
 public:
  GlobalRandFunctor() { op_ = CHECK_JUST(one::OpBuilder("uniform").Output("out").Build()); }
  Maybe<Tensor> operator()(const Shape& shape, const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp_tuple,
                           const Optional<Symbol<DType>>& dtype,
                           const Optional<one::Generator>& generator,
                           const bool& requires_grad) const {
    JUST(CheckDeviceIdsIsValid(placement));
    DataType dtype_val = DataType::kFloat;
    if (dtype.has_value()) {
      dtype_val = JUST(dtype)->data_type();
      if (dtype_val != DataType::kFloat && dtype_val != DataType::kDouble) {
        OF_UNIMPLEMENTED() << "Only support float and double in rand().";
      }
    }

    const auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);
    const auto& nd_sbp = JUST(GetNdSbp(sbp_tuple));

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("from", "to", "shape", "dtype", "seed", "nd_sbp");
    if (LazyMode::is_enabled()) {
      attrs.SetAllAttrs(static_cast<double>(0), static_cast<double>(1), shape, dtype_val,
                        static_cast<int64_t>(gen->current_seed()), *JUST(GetNdSbpStrList(nd_sbp)));
    } else {
      attrs.SetAllAttrs(static_cast<double>(0), static_cast<double>(1), shape, dtype_val,
                        static_cast<int64_t>(gen->current_seed()), NullOpt);
    }

    auto result = JUST(OpInterpUtil::Dispatch<Tensor>(
        *op_, {}, OpExprInterpContext(attrs, placement, nd_sbp, distribution_state)));
    JUST(result->set_requires_grad(requires_grad));
    return result;
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class RandNFunctor {
 public:
  RandNFunctor() { op_ = CHECK_JUST(one::OpBuilder("normal").Output("out").Build()); }
  Maybe<Tensor> operator()(const Shape& shape, const Optional<Symbol<DType>>& dtype,
                           const Optional<Symbol<Device>>& device,
                           const Optional<one::Generator>& generator,
                           const bool& requires_grad) const {
    DataType dtype_val = DataType::kFloat;
    if (dtype) { dtype_val = JUST(dtype)->data_type(); }
    if (dtype_val != DataType::kFloat && dtype_val != DataType::kDouble) {
      OF_UNIMPLEMENTED() << "Only support float and double in randn().";
    }

    const auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("mean", "std", "shape", "dtype", "seed");
    attrs.SetAllAttrs(static_cast<double>(0), static_cast<double>(1), shape, dtype_val,
                      static_cast<int64_t>(gen->current_seed()));

    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);

    OpExprInterpContext ctx(attrs, distribution_state);
    ctx.device = device;
    auto result = JUST(OpInterpUtil::Dispatch<Tensor>(*op_, {}, ctx));
    JUST(result->set_requires_grad(requires_grad));
    return result;
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class GlobalRandNFunctor {
 public:
  GlobalRandNFunctor() { op_ = CHECK_JUST(one::OpBuilder("normal").Output("out").Build()); }
  Maybe<Tensor> operator()(const Shape& shape, const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp_tuple,
                           const Optional<Symbol<DType>>& dtype,
                           const Optional<one::Generator>& generator,
                           const bool& requires_grad) const {
    JUST(CheckDeviceIdsIsValid(placement));
    DataType dtype_val = DataType::kFloat;
    if (dtype) { dtype_val = JUST(dtype)->data_type(); }
    if (dtype_val != DataType::kFloat && dtype_val != DataType::kDouble) {
      OF_UNIMPLEMENTED() << "Only support float and double in randn().";
    }

    const auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);
    const auto& nd_sbp = JUST(GetNdSbp(sbp_tuple));

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("mean", "std", "shape", "dtype", "seed", "nd_sbp");
    if (LazyMode::is_enabled()) {
      attrs.SetAllAttrs(static_cast<double>(0), static_cast<double>(1), shape, dtype_val,
                        static_cast<int64_t>(gen->current_seed()), *JUST(GetNdSbpStrList(nd_sbp)));
    } else {
      attrs.SetAllAttrs(static_cast<double>(0), static_cast<double>(1), shape, dtype_val,
                        static_cast<int64_t>(gen->current_seed()), NullOpt);
    }

    auto result = JUST(OpInterpUtil::Dispatch<Tensor>(
        *op_, {}, OpExprInterpContext(attrs, placement, nd_sbp, distribution_state)));
    JUST(result->set_requires_grad(requires_grad));
    return result;
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class RandIntFunctor {
 public:
  RandIntFunctor() { op_ = CHECK_JUST(one::OpBuilder("uniform_int").Output("out").Build()); }

  Maybe<Tensor> operator()(const int64_t low, const int64_t high, const Shape& shape,
                           const Optional<Symbol<DType>>& dtype,
                           const Optional<Symbol<Device>>& device,
                           const Optional<one::Generator>& generator,
                           const bool& requires_grad) const {
    DataType dtype_val = DataType::kInt64;
    if (dtype) { dtype_val = JUST(dtype)->data_type(); }

    const auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("shape", "from", "to", "dtype", "seed");
    attrs.SetAllAttrs(shape, low, high, dtype_val, static_cast<int64_t>(gen->current_seed()));

    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);

    OpExprInterpContext ctx(attrs, distribution_state);
    ctx.device = device;

    auto result = JUST(OpInterpUtil::Dispatch<Tensor>(*op_, {}, ctx));
    JUST(result->set_requires_grad(requires_grad));
    return result;
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class RandInt2Functor {
 public:
  Maybe<Tensor> operator()(const int64_t high, const Shape& shape,
                           const Optional<Symbol<DType>>& dtype,
                           const Optional<Symbol<Device>>& device,
                           const Optional<one::Generator>& generator,
                           const bool& requires_grad) const {
    return RandInt(/*low*/ 0, high, shape, dtype, device, generator, requires_grad);
  }
};

class RandIntLikeFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& input, const int64_t low,
                           const int64_t high, const Optional<Symbol<DType>>& dtype,
                           const Optional<Symbol<Device>>& device,
                           const Optional<one::Generator>& generator,
                           const bool& requires_grad) const {
    const Shape shape = *input->shape();
    return RandInt(low, high, shape, dtype, device, generator, requires_grad);
  }
};

class RandIntLike2Functor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& input, const int64_t high,
                           const Optional<Symbol<DType>>& dtype,
                           const Optional<Symbol<Device>>& device,
                           const Optional<one::Generator>& generator,
                           const bool& requires_grad) const {
    const Shape shape = *input->shape();
    return RandInt(/*low*/ 0, high, shape, dtype, device, generator, requires_grad);
  }
};

class GlobalRandIntFunctor {
 public:
  GlobalRandIntFunctor() { op_ = CHECK_JUST(one::OpBuilder("uniform_int").Output("out").Build()); }

  Maybe<Tensor> operator()(const int64_t low, const int64_t high, const Shape& shape,
                           const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp,
                           const Optional<Symbol<DType>>& dtype,
                           const Optional<one::Generator>& generator,
                           const bool& requires_grad) const {
    JUST(CheckDeviceIdsIsValid(placement));
    DataType dtype_val = DataType::kInt64;
    if (dtype) { dtype_val = JUST(dtype)->data_type(); }

    const auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);
    const auto& nd_sbp = JUST(GetNdSbp(sbp));

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("shape", "from", "to", "dtype", "seed", "nd_sbp");
    if (LazyMode::is_enabled()) {
      attrs.SetAllAttrs(shape, low, high, dtype_val, static_cast<int64_t>(gen->current_seed()),
                        *JUST(GetNdSbpStrList(nd_sbp)));
    } else {
      attrs.SetAllAttrs(shape, low, high, dtype_val, static_cast<int64_t>(gen->current_seed()),
                        NullOpt);
    }

    auto result = JUST(OpInterpUtil::Dispatch<Tensor>(
        *op_, {}, OpExprInterpContext(attrs, placement, nd_sbp, distribution_state)));

    JUST(result->set_requires_grad(requires_grad));
    return result;
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class GlobalRandInt2Functor {
 public:
  Maybe<Tensor> operator()(const int64_t high, const Shape& shape,
                           const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp,
                           const Optional<Symbol<DType>>& dtype,
                           const Optional<one::Generator>& generator,
                           const bool& requires_grad) const {
    JUST(CheckDeviceIdsIsValid(placement));
    return GlobalRandInt(/*low*/ 0, high, shape, placement, sbp, dtype, generator, requires_grad);
  }
};

class GlobalRandIntLikeFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& input, const int64_t low,
                           const int64_t high, const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp,
                           const Optional<Symbol<DType>>& dtype,
                           const Optional<one::Generator>& generator,
                           const bool& requires_grad) const {
    const Shape shape = *input->shape();
    return GlobalRandInt(low, high, shape, placement, sbp, dtype, generator, requires_grad);
  }
};

class GlobalRandIntLike2Functor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& input, const int64_t high,
                           const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp,
                           const Optional<Symbol<DType>>& dtype,
                           const Optional<one::Generator>& generator,
                           const bool& requires_grad) const {
    const Shape shape = *input->shape();
    return GlobalRandInt(/*low*/ 0, high, shape, placement, sbp, dtype, generator, requires_grad);
  }
};

class RandPermFunctor {
 public:
  RandPermFunctor() { randperm_op_ = CHECK_JUST(one::OpBuilder("randperm").Output("out").Build()); }
  Maybe<Tensor> operator()(const int32_t n, const Optional<one::Generator>& generator,
                           const Symbol<DType>& dtype, const Optional<Symbol<Device>>& device,
                           const bool& requires_grad) const {
    const auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("n", "seed");
    attrs.SetAllAttrs(n, static_cast<int64_t>(gen->current_seed()));

    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);

    OpExprInterpContext ctx(attrs, distribution_state);
    ctx.device = device;

    auto result = JUST(OpInterpUtil::Dispatch<Tensor>(*randperm_op_, {}, ctx));
    JUST(result->set_requires_grad(requires_grad));
    return functional::Cast(result, dtype, /*pin_memory=*/false);
  }

 private:
  std::shared_ptr<OpExpr> randperm_op_;
};

class GlobalRandPermFunctor {
 public:
  GlobalRandPermFunctor() {
    randperm_op_ = CHECK_JUST(one::OpBuilder("randperm").Output("out").Build());
  }
  Maybe<Tensor> operator()(const int32_t n, const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp_tuple,
                           const Optional<one::Generator>& generator, const Symbol<DType>& dtype,
                           const bool& requires_grad) const {
    JUST(CheckDeviceIdsIsValid(placement));
    const auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);
    const auto& nd_sbp = JUST(GetNdSbp(sbp_tuple));

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("n", "seed", "nd_sbp");
    if (LazyMode::is_enabled()) {
      attrs.SetAllAttrs(n, static_cast<int64_t>(gen->current_seed()),
                        *JUST(GetNdSbpStrList(nd_sbp)));
    } else {
      attrs.SetAllAttrs(n, static_cast<int64_t>(gen->current_seed()), NullOpt);
    }

    auto result = JUST(OpInterpUtil::Dispatch<Tensor>(
        *randperm_op_, {}, OpExprInterpContext(attrs, placement, nd_sbp, distribution_state)));

    JUST(result->set_requires_grad(requires_grad));
    return functional::Cast(result, dtype, /*pin_memory=*/false);
  }

 private:
  std::shared_ptr<OpExpr> randperm_op_;
};
}  // namespace impl

using namespace impl;

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<BernoulliFunctor>("Bernoulli");
  m.add_functor<BernoulliProbFunctor>("BernoulliProb");
  m.add_functor<RandPermFunctor>("RandPerm");
  m.add_functor<GlobalRandPermFunctor>("GlobalRandPerm");
  m.add_functor<RandFunctor>("Rand");
  m.add_functor<GlobalRandFunctor>("GlobalRand");
  m.add_functor<RandNFunctor>("RandN");
  m.add_functor<GlobalRandNFunctor>("GlobalRandN");
  m.add_functor<RandIntFunctor, RandInt2Functor>("RandInt");
  m.add_functor<GlobalRandIntFunctor, GlobalRandInt2Functor>("GlobalRandInt");
  m.add_functor<RandIntLikeFunctor, RandIntLike2Functor>("RandIntLike");
  m.add_functor<GlobalRandIntLikeFunctor, GlobalRandIntLike2Functor>("GlobalRandIntLike");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
