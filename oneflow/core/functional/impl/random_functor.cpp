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
#include "oneflow/core/framework/layout.h"
#include "oneflow/core/framework/mutable_attr_map.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/impl/unary_functor.h"
#include "oneflow/core/job/global_mode.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/user/kernels/distributions/common.h"
#include "oneflow/user/kernels/random_seed_util.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

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
                           const Optional<one::Generator>& generator, const bool& inplace) const {
    if (x->is_global()) { JUST(CheckDeviceIdsIsValid(JUST(x->parallel_desc()))); }
    auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    gen = JUST(GetGeneratorForLazyOrGlobal(gen, LazyMode::is_enabled(), x));
    auto& bernoulli_attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("dtype", "seed", "p");
    // p == -1 means bernoulli op doesn't use p to generate random number
    bernoulli_attrs.SetAllAttrs(dtype->data_type(), static_cast<int64_t>(gen->current_seed()),
                                static_cast<double>(-1));

    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);
    OpExprInterpContext ctx(bernoulli_attrs, distribution_state);
    if (inplace) {
      auto outputs = std::make_shared<TensorTuple>(1);
      JUST(CheckInplaceValid(x));
      (*outputs)[0] = x;
      JUST(OpInterpUtil::Dispatch(*bernoulli_op_, {x}, outputs.get(), ctx));
      return outputs->at(0);
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*bernoulli_op_, {x}, ctx);
    }
  }

 private:
  std::shared_ptr<OpExpr> bernoulli_op_;
};

class BernoulliInplaceFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Symbol<DType>& dtype,
                           const Optional<one::Generator>& generator) const {
    return Bernoulli(x, dtype, generator, true);
  }
};

class BernoulliProbFunctor {
 public:
  BernoulliProbFunctor() {
    bernoulli_op_ = CHECK_JUST(one::OpBuilder("bernoulli").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const double& p,
                           const Symbol<DType>& dtype, const Optional<one::Generator>& generator,
                           const bool& inplace) const {
    CHECK_OR_THROW(p >= 0.0 && p <= 1.0) << "bernoulli expects p to be in [0, 1], but got p=" << p;
    if (x->is_global()) { JUST(CheckDeviceIdsIsValid(JUST(x->parallel_desc()))); }

    auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    gen = JUST(GetGeneratorForLazyOrGlobal(gen, LazyMode::is_enabled(), x));
    auto& bernoulli_attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("dtype", "seed", "p");
    bernoulli_attrs.SetAllAttrs(dtype->data_type(), static_cast<int64_t>(gen->current_seed()), p);

    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);
    OpExprInterpContext ctx(bernoulli_attrs, distribution_state);
    if (inplace) {
      auto outputs = std::make_shared<TensorTuple>(1);
      JUST(CheckInplaceValid(x));
      (*outputs)[0] = x;
      JUST(OpInterpUtil::Dispatch(*bernoulli_op_, {x}, outputs.get(), ctx));
      return outputs->at(0);
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*bernoulli_op_, {x}, ctx);
    }
  }

 private:
  std::shared_ptr<OpExpr> bernoulli_op_;
};

class BernoulliProbInplaceFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const double& p,
                           const Symbol<DType>& dtype,
                           const Optional<one::Generator>& generator) const {
    return BernoulliProb(x, p, dtype, generator, true);
  }
};

class InplaceUniformFunctor {
 public:
  InplaceUniformFunctor() {
    uniform_op_ = CHECK_JUST(one::OpBuilder("uniform").Output("out").Build());
    uniform_int_op_ = CHECK_JUST(one::OpBuilder("uniform_int").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Scalar& from,
                           const Scalar& to) const {
    JUST(CheckInplaceValid(x));
    const Shape& shape = *(x->shape());
    std::shared_ptr<OpExpr> exec_op;
    const auto& dtype = x->dtype();
    bool IsInteger = false;

    if (dtype->is_floating_point()) {
      exec_op = uniform_op_;
    } else if (dtype->is_integer()) {
      exec_op = uniform_int_op_;
      IsInteger = true;
    } else {
      OF_UNIMPLEMENTED() << "Only support floating and int dtype.";
    }
    DataType dtype_val = dtype->data_type();

    Optional<Symbol<Device>> device;
    Optional<Symbol<ParallelDesc>> placement;
    Optional<Symbol<NdSbp>> nd_sbp;

    auto gen = JUST(one::DefaultAutoGenerator());
    if (x->is_global()) {
      JUST(CheckDeviceIdsIsValid(JUST(x->parallel_desc())));
      placement = JUST(x->parallel_desc());
      nd_sbp = JUST(x->nd_sbp());
      gen = JUST(GetGeneratorForLazyOrGlobal(gen, LazyMode::is_enabled(), placement, nd_sbp));
    } else {
      device = JUST(x->device());
      gen = JUST(GetGeneratorForLazyOrGlobal(gen, LazyMode::is_enabled(), NullOpt, NullOpt));
    }

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("from", "to", "shape", "dtype", "seed", "nd_sbp");
    Optional<std::vector<std::string>> attr_nd_sbp{NullOpt};
    if (nd_sbp) { attr_nd_sbp = *JUST(GetNdSbpStrList(JUST(nd_sbp))); }
    if (IsInteger) {
      attrs.SetAllAttrs(from.Value<int64_t>(), to.Value<int64_t>(), shape, dtype_val,
                        static_cast<int64_t>(gen->current_seed()), attr_nd_sbp);
    } else {
      attrs.SetAllAttrs(from.Value<double>(), to.Value<double>(), shape, dtype_val,
                        static_cast<int64_t>(gen->current_seed()), attr_nd_sbp);
    }

    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);
    OpExprInterpContext ctx(attrs, distribution_state);
    ctx.parallel_desc = placement;
    ctx.nd_sbp = nd_sbp;
    ctx.device = device;

    auto outputs = std::make_shared<TensorTuple>(1);
    (*outputs)[0] = x;
    JUST(OpInterpUtil::Dispatch(*exec_op, {}, outputs.get(), ctx));
    return outputs->at(0);
  }

 private:
  std::shared_ptr<OpExpr> uniform_op_;
  std::shared_ptr<OpExpr> uniform_int_op_;
};

class RandFunctor {
 public:
  RandFunctor() { op_ = CHECK_JUST(one::OpBuilder("uniform").Output("out").Build()); }
  Maybe<Tensor> operator()(const Shape& shape, const Optional<Symbol<DType>>& dtype,
                           const Optional<Symbol<Device>>& device,
                           const Optional<one::Generator>& generator,
                           const bool& requires_grad) const {
    if (GlobalMode::is_enabled()) {
      auto global_mode_gurad = GlobalMode::Guard(false);
      return JUST(functional::GlobalRand(shape, GetGlobalParallelDescFromDevice(device),
                                         *JUST(GetSbpList(GlobalMode::nd_sbp())), dtype, generator,
                                         requires_grad));
    }
    DataType dtype_val = GetDefaultDType()->data_type();
    if (dtype.has_value()) {
      dtype_val = JUST(dtype)->data_type();
      if (!JUST(dtype)->is_floating_point()) {
        OF_UNIMPLEMENTED() << "Only support floating dtype in rand().";
      }
    }

    auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    gen = JUST(GetGeneratorForLazyOrGlobal(gen, LazyMode::is_enabled(), NullOpt, NullOpt));

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
    DataType dtype_val = GetDefaultDType()->data_type();
    if (dtype.has_value()) {
      dtype_val = JUST(dtype)->data_type();
      if (dtype_val != DataType::kFloat && dtype_val != DataType::kDouble) {
        OF_UNIMPLEMENTED() << "Only support floating dtype in rand().";
      }
    }

    JUST(CheckDeviceIdsIsValid(placement));
    const auto& nd_sbp = JUST(GetNdSbp(sbp_tuple));
    auto attr_nd_sbp = *JUST(GetNdSbpStrList(nd_sbp));

    auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    gen = JUST(GetGeneratorForLazyOrGlobal(gen, LazyMode::is_enabled(), placement, nd_sbp));

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("from", "to", "shape", "dtype", "seed", "nd_sbp");
    attrs.SetAllAttrs(static_cast<double>(0), static_cast<double>(1), shape, dtype_val,
                      static_cast<int64_t>(gen->current_seed()), attr_nd_sbp);

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
  Maybe<Tensor> operator()(const Shape& shape, const Optional<Symbol<DType>>& dtype,
                           const Optional<Symbol<Device>>& device,
                           const Optional<one::Generator>& generator, const bool& requires_grad,
                           const Symbol<Layout>& layout) const {
    if (GlobalMode::is_enabled()) {
      auto global_mode_gurad = GlobalMode::Guard(false);
      return JUST(functional::GlobalRandN(shape, GetGlobalParallelDescFromDevice(device),
                                          *JUST(GetSbpList(GlobalMode::nd_sbp())), dtype, generator,
                                          requires_grad));
    }
    if (dtype.has_value() && !JUST(dtype)->is_floating_point()) {
      OF_UNIMPLEMENTED() << "Only support floating dtype in randn().";
    }
    const auto& out = Optional<one::Tensor>();
    return Normal(static_cast<double>(0), static_cast<double>(1), shape, out, dtype, device,
                  generator, requires_grad);
  }
};

class GlobalRandNFunctor {
 public:
  Maybe<Tensor> operator()(const Shape& shape, const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp_tuple,
                           const Optional<Symbol<DType>>& dtype,
                           const Optional<one::Generator>& generator,
                           const bool& requires_grad) const {
    if (dtype.has_value() && !JUST(dtype)->is_floating_point()) {
      OF_UNIMPLEMENTED() << "Only support floating dtype in randn().";
    }
    const auto& out = Optional<one::Tensor>();
    return GlobalNormal(static_cast<double>(0), static_cast<double>(1), shape, out, placement,
                        sbp_tuple, dtype, generator, requires_grad);
  }
};

class NormalFunctor {
 public:
  NormalFunctor() { op_ = CHECK_JUST(one::OpBuilder("normal").Output("out").Build()); }
  Maybe<Tensor> operator()(const float& mean, const float& std, const Shape& shape,
                           const Optional<one::Tensor>& out,
                           const Optional<Symbol<DType>>& optional_dtype,
                           const Optional<Symbol<Device>>& optional_device,
                           const Optional<one::Generator>& optional_generator,
                           const bool& requires_grad) const {
    Symbol<DType> dtype = GetDefaultDType();
    if (optional_dtype.has_value()) {
      if (!JUST(optional_dtype)->is_floating_point()) {
        OF_UNIMPLEMENTED() << "Only support float and double in normal().";
      }
      dtype = JUST(optional_dtype);
    }
    Symbol<Device> device = JUST(Device::New("cpu"));
    if (optional_device.has_value()) { device = JUST(optional_device); }

    if (out.has_value()) {
      auto out_tensor = JUST(out);
      Symbol<DType> output_tensor_dtype = out_tensor->dtype();
      if (optional_dtype.has_value()) {
        CHECK_OR_RETURN(output_tensor_dtype == dtype)
            << Error::RuntimeError() << "data type " << dtype->name()
            << " does not match data type of out parameter " << output_tensor_dtype->name();
      }
      dtype = output_tensor_dtype;
      Symbol<Device> out_tensor_device = JUST(out_tensor->device());
      if (optional_device.has_value()) {
        CHECK_OR_RETURN(out_tensor_device == JUST(optional_device))
            << Error::RuntimeError() << "device type " << device->ToString()
            << " does not match device type of out parameter " << out_tensor_device->ToString();
      }
      device = out_tensor_device;
    }

    auto gen = optional_generator.value_or(JUST(one::DefaultAutoGenerator()));
    gen = JUST(GetGeneratorForLazyOrGlobal(gen, LazyMode::is_enabled(), NullOpt, NullOpt));

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("mean", "std", "shape", "dtype", "seed");
    attrs.SetAllAttrs(static_cast<double>(mean), static_cast<double>(std), shape,
                      dtype->data_type(), static_cast<int64_t>(gen->current_seed()));

    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);
    OpExprInterpContext ctx(attrs, device, distribution_state);
    if (out.has_value()) {
      std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
      (*outputs)[0] = JUST(out);
      JUST(OpInterpUtil::Dispatch(*op_, {}, outputs.get(), ctx));
      return (*outputs)[0];
    }

    auto result = JUST(OpInterpUtil::Dispatch<Tensor>(*op_, {}, ctx));
    JUST(result->set_requires_grad(requires_grad));
    return result;
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class Normal2Functor {
 public:
  Maybe<Tensor> operator()(const float& mean, const float& std, const int32_t& shape,
                           const Optional<one::Tensor>& out,
                           const Optional<Symbol<DType>>& optional_dtype,
                           const Optional<Symbol<Device>>& optional_device,
                           const Optional<one::Generator>& optional_generator,
                           const bool& requires_grad) const {
    const Shape size = Shape({shape});
    return Normal(mean, std, size, out, optional_dtype, optional_device, optional_generator,
                  requires_grad);
  }
};

class GlobalNormalFunctor {
 public:
  GlobalNormalFunctor() { op_ = CHECK_JUST(one::OpBuilder("normal").Output("out").Build()); }
  Maybe<Tensor> operator()(const float& mean, const float& std, const Shape& shape,
                           const Optional<one::Tensor>& out, const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp_tuple,
                           const Optional<Symbol<DType>>& optional_dtype,
                           const Optional<one::Generator>& optional_generator,
                           const bool& requires_grad) const {
    Symbol<DType> dtype = DType::Float();
    if (optional_dtype.has_value()) {
      if (!JUST(optional_dtype)->is_floating_point()) {
        OF_UNIMPLEMENTED() << "Only support float and double in normal().";
      }
      dtype = JUST(optional_dtype);
    }

    if (out.has_value()) {
      auto out_tensor = JUST(out);
      Symbol<DType> output_tensor_dtype = out_tensor->dtype();
      if (optional_dtype.has_value()) {
        CHECK_OR_RETURN(output_tensor_dtype == dtype)
            << Error::RuntimeError() << "data type " << dtype->name()
            << " does not match data type of out parameter (" << output_tensor_dtype->name();
      }
      dtype = output_tensor_dtype;
    }

    JUST(CheckDeviceIdsIsValid(placement));
    const auto& nd_sbp = JUST(GetNdSbp(sbp_tuple));
    auto attr_nd_sbp = *JUST(GetNdSbpStrList(nd_sbp));

    std::shared_ptr<Generator> gen = optional_generator.value_or(JUST(one::DefaultAutoGenerator()));
    gen = JUST(GetGeneratorForLazyOrGlobal(gen, LazyMode::is_enabled(), placement, nd_sbp));

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("mean", "std", "shape", "dtype", "seed", "nd_sbp");
    attrs.SetAllAttrs(static_cast<double>(mean), static_cast<double>(std), shape,
                      dtype->data_type(), static_cast<int64_t>(gen->current_seed()), attr_nd_sbp);

    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);
    OpExprInterpContext ctx(attrs, placement, nd_sbp, distribution_state);
    if (out.has_value()) {
      std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
      (*outputs)[0] = JUST(out);
      JUST(OpInterpUtil::Dispatch(*op_, {}, outputs.get(), ctx));
      return (*outputs)[0];
    }

    auto result = JUST(OpInterpUtil::Dispatch<Tensor>(*op_, {}, ctx));
    JUST(result->set_requires_grad(requires_grad));
    return result;
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class GlobalNormal2Functor {
 public:
  Maybe<Tensor> operator()(const float& mean, const float& std, const int32_t& shape,
                           const Optional<one::Tensor>& out, const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp_tuple,
                           const Optional<Symbol<DType>>& optional_dtype,
                           const Optional<one::Generator>& optional_generator,
                           const bool& requires_grad) const {
    const Shape size = Shape({shape});
    return GlobalNormal(mean, std, size, out, placement, sbp_tuple, optional_dtype,
                        optional_generator, requires_grad);
  }
};

class RandnLikeFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& input,
                           const Optional<Symbol<DType>>& dtype,
                           const Optional<Symbol<Device>>& device,
                           const Optional<one::Generator>& generator,
                           const bool& requires_grad) const {
    return RandN(*input->shape(), dtype.value_or(input->dtype()),
                 device.value_or(JUST(input->device())), generator, requires_grad,
                 Layout::Strided());
  }
};

class GlobalRandnLikeFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& input,
                           const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp,
                           const Optional<Symbol<DType>>& dtype,
                           const Optional<one::Generator>& generator,
                           const bool& requires_grad) const {
    return GlobalRandN(*input->shape(), placement, sbp, dtype.value_or(input->dtype()), generator,
                       requires_grad);
  }
};

class RandIntFunctor {
 public:
  RandIntFunctor() { op_ = CHECK_JUST(one::OpBuilder("uniform_int").Output("out").Build()); }

  Maybe<Tensor> operator()(const int64_t low, const int64_t high, const Shape& shape,
                           const Optional<Symbol<DType>>& dtype,
                           const Optional<Symbol<Device>>& device,
                           const Optional<one::Generator>& generator,
                           const bool& requires_grad) const {
    if (GlobalMode::is_enabled()) {
      auto global_mode_gurad = GlobalMode::Guard(false);
      return JUST(functional::GlobalRandInt(
          low, high, shape, GetGlobalParallelDescFromDevice(device),
          *JUST(GetSbpList(GlobalMode::nd_sbp())), dtype, generator, requires_grad));
    }

    DataType dtype_val = DataType::kInt64;
    if (dtype) { dtype_val = JUST(dtype)->data_type(); }

    auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    gen = JUST(GetGeneratorForLazyOrGlobal(gen, LazyMode::is_enabled(), NullOpt, NullOpt));

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

    const auto& nd_sbp = JUST(GetNdSbp(sbp));
    auto attr_nd_sbp = *JUST(GetNdSbpStrList(nd_sbp));

    auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    gen = JUST(GetGeneratorForLazyOrGlobal(gen, LazyMode::is_enabled(), placement, nd_sbp));

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("shape", "from", "to", "dtype", "seed", "nd_sbp");
    attrs.SetAllAttrs(shape, low, high, dtype_val, static_cast<int64_t>(gen->current_seed()),
                      attr_nd_sbp);

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
  Maybe<Tensor> operator()(const int32_t n, const Optional<one::Generator>& generator,
                           const Symbol<DType>& dtype, const Optional<Symbol<Device>>& device,
                           const bool& requires_grad) const {
    if (GlobalMode::is_enabled()) {
      auto global_mode_gurad = GlobalMode::Guard(false);
      return JUST(functional::GlobalRandPerm(n, GetGlobalParallelDescFromDevice(device),
                                             *JUST(GetSbpList(GlobalMode::nd_sbp())), generator,
                                             dtype, requires_grad));
    }

    auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    gen = JUST(GetGeneratorForLazyOrGlobal(gen, LazyMode::is_enabled(), NullOpt, NullOpt));

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
    const auto& nd_sbp = JUST(GetNdSbp(sbp_tuple));
    auto attr_nd_sbp = *JUST(GetNdSbpStrList(nd_sbp));

    auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    gen = JUST(GetGeneratorForLazyOrGlobal(gen, LazyMode::is_enabled(), placement, nd_sbp));

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("n", "seed", "nd_sbp");
    attrs.SetAllAttrs(n, static_cast<int64_t>(gen->current_seed()), attr_nd_sbp);

    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);
    auto result = JUST(OpInterpUtil::Dispatch<Tensor>(
        *randperm_op_, {}, OpExprInterpContext(attrs, placement, nd_sbp, distribution_state)));
    JUST(result->set_requires_grad(requires_grad));
    return functional::Cast(result, dtype, /*pin_memory=*/false);
  }

 private:
  std::shared_ptr<OpExpr> randperm_op_;
};

class ExponentialFunctor {
 public:
  ExponentialFunctor() { op_ = CHECK_JUST(one::OpBuilder("exponential").Output("out").Build()); }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const float& lambd,
                           const Optional<one::Generator>& generator) const {
    DataType dtype_val = x->dtype()->data_type();

    Optional<Symbol<Device>> device;
    Optional<Symbol<ParallelDesc>> placement;
    Optional<Symbol<NdSbp>> nd_sbp;

    auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    if (x->is_global()) {
      JUST(CheckDeviceIdsIsValid(JUST(x->parallel_desc())));
      placement = JUST(x->parallel_desc());
      nd_sbp = JUST(x->nd_sbp());
      gen = JUST(GetGeneratorForLazyOrGlobal(gen, LazyMode::is_enabled(), placement, nd_sbp));
    } else {
      device = JUST(x->device());
      gen = JUST(GetGeneratorForLazyOrGlobal(gen, LazyMode::is_enabled(), NullOpt, NullOpt));
    }

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("seed", "lambd", "dtype", "out_shape", "nd_sbp");
    const Shape& out_shape = *(x->shape());
    Optional<std::vector<std::string>> attr_nd_sbp{NullOpt};
    if (nd_sbp) { attr_nd_sbp = *JUST(GetNdSbpStrList(JUST(nd_sbp))); }
    attrs.SetAllAttrs(static_cast<int64_t>(gen->current_seed()), lambd, dtype_val, out_shape,
                      attr_nd_sbp);

    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);
    OpExprInterpContext ctx(attrs, distribution_state);
    ctx.device = device;
    ctx.parallel_desc = placement;
    ctx.nd_sbp = nd_sbp;

    std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
    outputs->at(0) = x;
    JUST(OpInterpUtil::Dispatch(*op_, {}, outputs.get(), ctx));
    return outputs->at(0);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

// NOTE(Liang Depeng): The implementation of MultinomialFunctor is modified from
//                    https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Distributions.cpp#L548
class MultinomialFunctor {
 public:
  MultinomialFunctor() {
    op_cpu_ =
        CHECK_JUST(one::OpBuilder("multinomial_with_replacement").Input("x").Output("out").Build());
    op_gpu_ = CHECK_JUST(one::OpBuilder("multinomial_with_replacement")
                             .Input("x")
                             .Input("prefix_sum")
                             .Output("out")
                             .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const int& num_samples,
                           const bool& replacement,
                           const Optional<one::Generator>& generator) const {
    CHECK_OR_RETURN(x->ndim() > 0 && x->ndim() <= 2)
        << "The input probability tensor must be 1 or 2 dim, "
        << "but got: " << x->ndim();
    CHECK_OR_RETURN(x->dtype()->is_floating_point())
        << "multinomial only supports floating-point dtypes for input, but got: "
        << x->dtype()->name();
    CHECK_OR_RETURN(num_samples > 0) << "cannot sample num_samples <= 0 samples";
    int64_t num_categories = x->dim(x->ndim() - 1);
    CHECK_OR_RETURN(replacement || num_samples <= num_categories)
        << "cannot sample num_samples > prob_dist.size(-1) samples without replacement";

    /* The largest consecutive integer representable in float32 (2^24) */
    constexpr int64_t FLOAT32_MAX_CONSECUTIVE_INT = 1 << (FLT_MANT_DIG);
    // Since the index tensor is float, numCategories cannot exceed max float integer precision
    CHECK_OR_RETURN(num_categories <= FLOAT32_MAX_CONSECUTIVE_INT)
        << "number of categories cannot exceed 2^24";

    // Fast-path for no replacement.
    // Reference:
    // https://github.com/pytorch/pytorch/issues/11931#issuecomment-625882503
    if (!replacement) {
      // The algorithm is from gumbel softmax.
      // s = argmax( logp - log(-log(eps)) ) where eps ~ U(0, 1)
      // Here we can apply exp to the formula which will not affect result of
      // argmax or topk. Then we have
      // s = argmax( p / (-log(eps)) ) where eps ~ U(0, 1).
      // We can also simplify the formula above by
      // s = argmax( p / q ) where q ~ Exp(1)
      std::shared_ptr<Tensor> q =
          JUST(functional::Empty(*(x->shape()), x->dtype(), JUST(x->device()),
                                 /*requires_grad=*/x->requires_grad(), /*pin_memory=*/false));
      q = JUST(functional::Exponential(q, 1, generator));
      // In theory the probability to generate 0 from exponential distribution is
      // 0. However, on CUDA side there is a protection to avoid 0s, but on CPU
      // side, there is a very low probability to generate 0 from
      // exponential<double>. The probability is about 2^(-DBL_MANT_DIG). We just
      // ignore it here, but there may be some risk to get invalid output on CPU.
      q = JUST(functional::Div(x, q));
      std::shared_ptr<Tensor> result;
      if (num_samples == 1) {
        result = JUST(functional::ArgMax(q, -1, true, JUST(DType::Get(DataType::kInt64))));
      } else {
        std::shared_ptr<TensorTuple> temp =
            JUST(functional::TopK(q, num_samples, -1,
                                  /*largest=*/true, /*sorted=*/true));
        result = (*temp)[1];
      }
      return result;
    }

    DeviceType input_device = DeviceType::kCPU;
    if (x->is_global()) {
      JUST(CheckDeviceIdsIsValid(JUST(x->parallel_desc())));
      input_device = JUST(x->parallel_desc())->device_type();
    } else {
      input_device = JUST(x->device())->enum_type();
    }
    auto gen = generator.value_or(JUST(one::DefaultAutoGenerator()));
    gen = JUST(GetGeneratorForLazyOrGlobal(gen, LazyMode::is_enabled(), x));
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("seed", "num_samples");
    attrs.SetAllAttrs(static_cast<int64_t>(gen->current_seed()), num_samples);

    const auto& distribution_state = std::make_shared<DistributionKernelState>(gen);
    OpExprInterpContext ctx(attrs, distribution_state);

    if (input_device == DeviceType::kCPU) {
      return OpInterpUtil::Dispatch<Tensor>(*op_cpu_, {x}, ctx);
    } else {
      std::shared_ptr<Tensor> sum_last_dim = JUST(functional::ReduceSum(x, {-1}, true));
      std::shared_ptr<Tensor> norm_dist = JUST(functional::Div(x, sum_last_dim));
      std::shared_ptr<Tensor> prefix_sum = JUST(functional::Cumsum(norm_dist, -1, x->dtype()));
      return OpInterpUtil::Dispatch<Tensor>(*op_gpu_, {norm_dist, prefix_sum}, ctx);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_cpu_;
  std::shared_ptr<OpExpr> op_gpu_;
};

}  // namespace impl

using namespace impl;

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<BernoulliFunctor>("Bernoulli");
  m.add_functor<BernoulliInplaceFunctor>("BernoulliInplace");
  m.add_functor<BernoulliProbFunctor>("BernoulliProb");
  m.add_functor<BernoulliProbInplaceFunctor>("BernoulliProbInplace");
  m.add_functor<RandPermFunctor>("RandPerm");
  m.add_functor<GlobalRandPermFunctor>("GlobalRandPerm");
  m.add_functor<RandFunctor>("Rand");
  m.add_functor<GlobalRandFunctor>("GlobalRand");
  m.add_functor<RandNFunctor>("RandN");
  m.add_functor<GlobalRandNFunctor>("GlobalRandN");
  m.add_functor<impl::NormalFunctor>("Normal");
  m.add_functor<impl::Normal2Functor>("Normal2");
  m.add_functor<impl::GlobalNormalFunctor>("GlobalNormal");
  m.add_functor<impl::GlobalNormal2Functor>("GlobalNormal2");
  m.add_functor<RandnLikeFunctor>("RandnLike");
  m.add_functor<GlobalRandnLikeFunctor>("GlobalRandnLike");
  m.add_functor<RandIntFunctor, RandInt2Functor>("RandInt");
  m.add_functor<GlobalRandIntFunctor, GlobalRandInt2Functor>("GlobalRandInt");
  m.add_functor<RandIntLikeFunctor, RandIntLike2Functor>("RandIntLike");
  m.add_functor<GlobalRandIntLikeFunctor, GlobalRandIntLike2Functor>("GlobalRandIntLike");
  m.add_functor<ExponentialFunctor>("Exponential");
  m.add_functor<MultinomialFunctor>("Multinomial");
  m.add_functor<InplaceUniformFunctor>("InplaceUniform");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
