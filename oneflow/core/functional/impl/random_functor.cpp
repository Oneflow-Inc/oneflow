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
#include "oneflow/core/common/multi_client.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_generated.h"
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
    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);
    auto ctx = std::make_shared<schema::BernoulliOp>();
    ctx->set_dtype(dtype->data_type());
    ctx->set_seed(gen->current_seed());
    ctx->state = distribution_state;
    return OpInterpUtil::Dispatch<Tensor>(*bernoulli_op_, {x}, ctx);
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
    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);

    auto ctx = std::make_shared<schema::UniformOp>();
    ctx->set_from(0);
    ctx->set_to(1);
    ctx->set_shape(shape);
    ctx->set_dtype(dtype_val);
    ctx->set_seed(gen->current_seed());
    ctx->device = device;
    ctx->state = distribution_state;
    auto result = JUST(OpInterpUtil::Dispatch<Tensor>(*op_, {}, ctx));
    JUST(result->set_requires_grad(requires_grad));
    return result;
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ConsistentRandFunctor {
 public:
  ConsistentRandFunctor() { op_ = CHECK_JUST(one::OpBuilder("uniform").Output("out").Build()); }
  Maybe<Tensor> operator()(const Shape& shape, const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<cfg::SbpParallel>>& sbp_tuple,
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
    auto ctx = std::make_shared<schema::UniformOp>();
    ctx->set_from(0);
    ctx->set_to(1);
    ctx->set_shape(shape);
    ctx->set_dtype(dtype_val);
    ctx->set_seed(gen->current_seed());
    ctx->set_nd_sbp(*JUST(NdSbpToString(JUST(ctx->sbp))));

    ctx->state = distribution_state;
    ctx->parallel_desc = placement;
    ctx->sbp = JUST(GetNdSbp(sbp_tuple));
    auto result = JUST(OpInterpUtil::Dispatch<Tensor>(*op_, {}, ctx));
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
    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);

    auto ctx = std::make_shared<schema::NormalOp>();
    ctx->set_mean(0);
    ctx->set_std(1);
    ctx->set_shape(shape);
    ctx->set_dtype(dtype_val);
    ctx->set_seed(gen->current_seed());
    ctx->state = distribution_state;
    ctx->device = device;
    auto result = JUST(OpInterpUtil::Dispatch<Tensor>(*op_, {}, ctx));
    JUST(result->set_requires_grad(requires_grad));
    return result;
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ConsistentRandNFunctor {
 public:
  ConsistentRandNFunctor() { op_ = CHECK_JUST(one::OpBuilder("normal").Output("out").Build()); }
  Maybe<Tensor> operator()(const Shape& shape, const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<cfg::SbpParallel>>& sbp_tuple,
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

    auto ctx = std::make_shared<schema::NormalOp>();
    ctx->set_mean(0);
    ctx->set_std(1);
    ctx->set_shape(shape);
    ctx->set_dtype(dtype_val);
    ctx->set_seed(gen->current_seed());
    ctx->set_nd_sbp(*JUST(NdSbpToString(JUST(ctx->sbp))));

    ctx->state = distribution_state;
    ctx->parallel_desc = placement;
    ctx->sbp = JUST(GetNdSbp(sbp_tuple));
    auto result = JUST(OpInterpUtil::Dispatch<Tensor>(*op_, {}, ctx));
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
    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);
    auto ctx = std::make_shared<schema::UniformIntOp>();
    ctx->set_shape(shape);
    ctx->set_from(low);
    ctx->set_to(high);
    ctx->set_dtype(dtype_val);
    ctx->set_seed(gen->current_seed());
    ctx->state = distribution_state;
    ctx->device = device;
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

class ConsistentRandIntFunctor {
 public:
  ConsistentRandIntFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("uniform_int").Output("out").Build());
  }

  Maybe<Tensor> operator()(const int64_t low, const int64_t high, const Shape& shape,
                           const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<cfg::SbpParallel>>& sbp_tuple,
                           const Optional<Symbol<DType>>& dtype,
                           const Optional<one::Generator>& generator,
                           const bool& requires_grad) const {
    JUST(CheckDeviceIdsIsValid(placement));
    DataType dtype_val = DataType::kInt64;
    if (dtype) { dtype_val = JUST(dtype)->data_type(); }
    const auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);
    auto ctx = std::make_shared<schema::UniformIntOp>();
    ctx->set_shape(shape);
    ctx->set_from(low);
    ctx->set_to(high);
    ctx->set_dtype(dtype_val);
    ctx->set_seed(gen->current_seed());
    ctx->set_nd_sbp(*JUST(NdSbpToString(JUST(ctx->sbp))));

    ctx->state = distribution_state;
    ctx->parallel_desc = placement;
    ctx->sbp = JUST(GetNdSbp(sbp_tuple));
    auto result = JUST(OpInterpUtil::Dispatch<Tensor>(*op_, {}, ctx));
    JUST(result->set_requires_grad(requires_grad));
    return result;
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ConsistentRandInt2Functor {
 public:
  Maybe<Tensor> operator()(const int64_t high, const Shape& shape,
                           const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<cfg::SbpParallel>>& sbp_tuple,
                           const Optional<Symbol<DType>>& dtype,
                           const Optional<one::Generator>& generator,
                           const bool& requires_grad) const {
    JUST(CheckDeviceIdsIsValid(placement));
    return ConsistentRandInt(/*low*/ 0, high, shape, placement, sbp_tuple, dtype, generator,
                             requires_grad);
  }
};

class RandPermFunctor {
 public:
  RandPermFunctor() { randperm_op_ = CHECK_JUST(one::OpBuilder("randperm").Output("out").Build()); }
  Maybe<Tensor> operator()(const int32_t n, const Optional<one::Generator>& generator,
                           const Symbol<DType>& dtype, const Optional<Symbol<Device>>& device,
                           const bool& requires_grad) const {
    const auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);
    auto ctx = std::make_shared<schema::RandpermOp>();
    ctx->set_n(n);
    ctx->set_seed(gen->current_seed());
    ctx->state = distribution_state;
    ctx->device = device;

    auto result = JUST(OpInterpUtil::Dispatch<Tensor>(*randperm_op_, {}, ctx));
    JUST(result->set_requires_grad(requires_grad));
    return result;
  }

 private:
  std::shared_ptr<OpExpr> randperm_op_;
};

class ConsistentRandPermFunctor {
 public:
  ConsistentRandPermFunctor() {
    randperm_op_ = CHECK_JUST(one::OpBuilder("randperm").Output("out").Build());
  }
  Maybe<Tensor> operator()(const int32_t n, const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<cfg::SbpParallel>>& sbp_tuple,
                           const Optional<one::Generator>& generator, const Symbol<DType>& dtype,
                           const bool& requires_grad) const {
    JUST(CheckDeviceIdsIsValid(placement));
    const auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);
    auto ctx = std::make_shared<schema::RandpermOp>();
    ctx->set_n(n);
    ctx->set_seed(gen->current_seed());
    ctx->set_nd_sbp(*JUST(NdSbpToString(JUST(ctx->sbp))));

    ctx->state = distribution_state;
    ctx->parallel_desc = placement;
    ctx->sbp = JUST(GetNdSbp(sbp_tuple));
    auto result = JUST(OpInterpUtil::Dispatch<Tensor>(*randperm_op_, {}, ctx));
    JUST(result->set_requires_grad(requires_grad));
    return result;
  }

 private:
  std::shared_ptr<OpExpr> randperm_op_;
};
}  // namespace impl

using namespace impl;

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<BernoulliFunctor>("Bernoulli");
  m.add_functor<RandPermFunctor>("RandPerm");
  m.add_functor<ConsistentRandPermFunctor>("ConsistentRandPerm");
  m.add_functor<RandFunctor>("Rand");
  m.add_functor<ConsistentRandFunctor>("ConsistentRand");
  m.add_functor<RandNFunctor>("RandN");
  m.add_functor<ConsistentRandNFunctor>("ConsistentRandN");
  m.add_functor<RandIntFunctor, RandInt2Functor>("RandInt");
  m.add_functor<ConsistentRandIntFunctor, ConsistentRandInt2Functor>("ConsistentRandInt");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
