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
#include "oneflow/core/common/functor_util.h"
#include "oneflow/core/framework/attr_map.h"
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
  struct Bernoulli {
    Maybe<AttrMap> operator()(DataType dtype, int64_t seed) {
      MutableAttrMap bernoulli_attrs;
      JUST(bernoulli_attrs.SetAttr<DataType>("dtype", dtype));
      JUST(bernoulli_attrs.SetAttr<int64_t>("seed", seed));
      return AttrMap(bernoulli_attrs);
    }
  };
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Symbol<DType>& dtype,
                           const Optional<one::Generator>& generator) const {
    const auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(Bernoulli);
    const auto bernoulli_attrs = *JUST(GetAttrs(dtype->data_type(), gen->current_seed()));
    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);

    return OpInterpUtil::Dispatch<Tensor>(*bernoulli_op_, {x},
                                          OpExprInterpContext(bernoulli_attrs, distribution_state));
  }

 private:
  std::shared_ptr<OpExpr> bernoulli_op_;
};

class RandFunctor {
 public:
  RandFunctor() { op_ = CHECK_JUST(one::OpBuilder("uniform").Output("out").Build()); }
  struct Rand {
    Maybe<AttrMap> operator()(const Shape& shape, DataType dtype, int64_t seed) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<double>("from", 0));
      JUST(attrs.SetAttr<double>("to", 1));
      JUST(attrs.SetAttr<Shape>("shape", shape));
      JUST(attrs.SetAttr<DataType>("dtype", dtype));
      JUST(attrs.SetAttr<int64_t>("seed", seed));
      return AttrMap(attrs);
    }
  };
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
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(Rand);
    const auto attrs = *JUST(GetAttrs(shape, dtype_val, gen->current_seed()));
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
  struct GlobalRand {
    Maybe<AttrMap> operator()(const Shape& shape, DataType dtype, int64_t seed,
                              bool is_lazy_mode_enabled, Symbol<NdSbp> nd_sbp) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<double>("from", 0));
      JUST(attrs.SetAttr<double>("to", 1));
      JUST(attrs.SetAttr<Shape>("shape", shape));
      JUST(attrs.SetAttr<DataType>("dtype", dtype));
      JUST(attrs.SetAttr<int64_t>("seed", seed));
      if (is_lazy_mode_enabled) {
        JUST(attrs.SetAttr<std::vector<std::string>>("nd_sbp", *JUST(GetNdSbpStrList(nd_sbp))));
      }
      return AttrMap(attrs);
    }
  };
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

    const auto& nd_sbp = JUST(GetNdSbp(sbp_tuple));
    const auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(GlobalRand);
    const auto attrs =
        *JUST(GetAttrs(shape, dtype_val, gen->current_seed(), LazyMode::is_enabled(), nd_sbp));
    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);
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
  struct RandN {
    Maybe<AttrMap> operator()(const Shape& shape, DataType dtype, int64_t seed) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<double>("from", 0));
      JUST(attrs.SetAttr<double>("to", 1));
      JUST(attrs.SetAttr<Shape>("shape", shape));
      JUST(attrs.SetAttr<DataType>("dtype", dtype));
      JUST(attrs.SetAttr<int64_t>("seed", seed));
      return AttrMap(attrs);
    }
  };
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
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(RandN);
    const auto attrs = *JUST(GetAttrs(shape, dtype_val, gen->current_seed()));
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
  struct GlobalRandN {
    Maybe<AttrMap> operator()(const Shape& shape, DataType dtype, int64_t seed,
                              bool is_lazy_mode_enabled, Symbol<NdSbp> nd_sbp) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<double>("from", 0));
      JUST(attrs.SetAttr<double>("to", 1));
      JUST(attrs.SetAttr<Shape>("shape", shape));
      JUST(attrs.SetAttr<DataType>("dtype", dtype));
      JUST(attrs.SetAttr<int64_t>("seed", seed));
      if (is_lazy_mode_enabled) {
        JUST(attrs.SetAttr<std::vector<std::string>>("nd_sbp", *JUST(GetNdSbpStrList(nd_sbp))));
      }
      return AttrMap(attrs);
    }
  };
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
    const auto& nd_sbp = JUST(GetNdSbp(sbp_tuple));
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(GlobalRandN);
    const auto attrs =
        *JUST(GetAttrs(shape, dtype_val, gen->current_seed(), LazyMode::is_enabled(), nd_sbp));
    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);
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

  struct RandInt {
    Maybe<AttrMap> operator()(const Shape& shape, int64_t low, int64_t high, DataType dtype,
                              int64_t seed) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<Shape>("shape", shape));
      JUST(attrs.SetAttr<int64_t>("from", low));
      JUST(attrs.SetAttr<int64_t>("to", high));
      JUST(attrs.SetAttr<DataType>("dtype", dtype));
      JUST(attrs.SetAttr<int64_t>("seed", seed));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const int64_t low, const int64_t high, const Shape& shape,
                           const Optional<Symbol<DType>>& dtype,
                           const Optional<Symbol<Device>>& device,
                           const Optional<one::Generator>& generator,
                           const bool& requires_grad) const {
    DataType dtype_val = DataType::kInt64;
    if (dtype) { dtype_val = JUST(dtype)->data_type(); }

    const auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(RandInt);
    const auto attrs = *JUST(GetAttrs(shape, low, high, dtype_val, gen->current_seed()));

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

  struct GlobalRandInt {
    Maybe<AttrMap> operator()(const Shape& shape, int64_t low, int64_t high, DataType dtype,
                              int64_t seed, bool is_lazy_mode, Symbol<NdSbp> nd_sbp) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<Shape>("shape", shape));
      JUST(attrs.SetAttr<int64_t>("from", low));
      JUST(attrs.SetAttr<int64_t>("to", high));
      JUST(attrs.SetAttr<DataType>("dtype", dtype));
      JUST(attrs.SetAttr<int64_t>("seed", seed));
      if (is_lazy_mode) {
        JUST(attrs.SetAttr<std::vector<std::string>>("nd_sbp", *JUST(GetNdSbpStrList(nd_sbp))));
      }
      return AttrMap(attrs);
    }
  };

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
    const auto& nd_sbp = JUST(GetNdSbp(sbp));
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(GlobalRandInt);
    const auto attrs = *JUST(
        GetAttrs(shape, low, high, dtype_val, gen->current_seed(), LazyMode::is_enabled(), nd_sbp));

    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);

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
  struct RandPerm {
    Maybe<AttrMap> operator()(int32_t n, int64_t seed) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<int32_t>("n", n));
      JUST(attrs.SetAttr<int64_t>("seed", seed));
      return AttrMap(attrs);
    }
  };
  Maybe<Tensor> operator()(const int32_t n, const Optional<one::Generator>& generator,
                           const Symbol<DType>& dtype, const Optional<Symbol<Device>>& device,
                           const bool& requires_grad) const {
    const auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(RandPerm);
    const auto attrs = *JUST(GetAttrs(n, gen->current_seed()));

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
  struct GlobalRandPerm {
    Maybe<AttrMap> operator()(int32_t n, int64_t seed, bool is_lazy_mode, Symbol<NdSbp> nd_sbp) {
      MutableAttrMap attrs;
      JUST(attrs.SetAttr<int32_t>("n", n));
      JUST(attrs.SetAttr<int64_t>("seed", seed));
      if (is_lazy_mode) {
        JUST(attrs.SetAttr<std::vector<std::string>>("nd_sbp", *JUST(GetNdSbpStrList(nd_sbp))));
      }
      return AttrMap(attrs);
    }
  };
  Maybe<Tensor> operator()(const int32_t n, const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp_tuple,
                           const Optional<one::Generator>& generator, const Symbol<DType>& dtype,
                           const bool& requires_grad) const {
    JUST(CheckDeviceIdsIsValid(placement));
    const auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    const auto& nd_sbp = JUST(GetNdSbp(sbp_tuple));
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(GlobalRandPerm);
    const auto attrs = *JUST(GetAttrs(n, gen->current_seed(), LazyMode::is_enabled(), nd_sbp));
    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);
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
