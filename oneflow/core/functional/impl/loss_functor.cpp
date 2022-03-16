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
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/functional/sequence_function.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/job/sbp_parallel.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class LossFunctorBase {
 public:
  Maybe<Tensor> apply_reduction(const Maybe<Tensor>& x, const std::string& reduction) const {
    CHECK_OR_RETURN(reduction == "none" || reduction == "sum" || reduction == "mean")
        << "Reduction should be none, sum or mean.";
    if (reduction == "sum") { return functional::ReduceSum(JUST(x), {}, false); }
    if (reduction == "mean") { return functional::ReduceMean(JUST(x), {}, false); }
    return x;
  }

 protected:
  LossFunctorBase() = default;
  virtual ~LossFunctorBase() = default;
};

class MseLossFunctor : public LossFunctorBase {
 public:
  MseLossFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const std::string& reduction) const {
    const auto out =
        sequence_function(functional::Sub).then(functional::Square).call(input, target);
    return apply_reduction(out, reduction);
  }
};

class L1LossFunctor : public LossFunctorBase {
 public:
  L1LossFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const std::string& reduction) const {
    const auto out = sequence_function(functional::Sub).then(functional::Abs).call(input, target);
    return apply_reduction(out, reduction);
  }
};

class SmoothL1LossFunctor : LossFunctorBase {
 public:
  SmoothL1LossFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("smooth_l1_loss").Input("input").Input("target").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target, const float& beta,
                           const std::string& reduction) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("beta", beta));
    return apply_reduction(OpInterpUtil::Dispatch<Tensor>(*op_, {input, target}, attrs), reduction);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class SmoothL1LossGradFunctor {
 public:
  SmoothL1LossGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("smooth_l1_loss_grad")
                         .Input("dy")
                         .Input("input")
                         .Input("target")
                         .Output("dx")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target, const float& beta) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("beta", beta));

    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {dy, input, target}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class KLDivLossFunctor : public LossFunctorBase {
 public:
  KLDivLossFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("kl_div_loss").Input("input").Input("target").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target, const bool log_target,
                           const std::string& reduction) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<bool>("log_target", log_target));
    if (reduction != "batchmean") {
      return apply_reduction(OpInterpUtil::Dispatch<Tensor>(*op_, {input, target}, attrs),
                             reduction);
    }
    return SequenceFunction<Maybe<Tensor>()>([&, this]() {
             return apply_reduction(OpInterpUtil::Dispatch<Tensor>(*op_, {input, target}, attrs),
                                    "sum");
           })
        .then_if(input->shape()->NumAxes() != 0,
                 [&](const auto& x) { return functional::ScalarDiv(x, input->shape()->At(0)); })
        .call();
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class KLDivLossInputGradFunctor {
 public:
  KLDivLossInputGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("kl_div_loss_input_grad")
                         .Input("input")
                         .Input("target")
                         .Input("dy")
                         .Output("dx")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const bool log_target) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<bool>("log_target", log_target));

    return OpInterpUtil::Dispatch<Tensor>(*op_, {input, target, dy}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class KLDivLossTargetGradFunctor {
 public:
  KLDivLossTargetGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("kl_div_loss_target_grad")
                         .Input("input")
                         .Input("target")
                         .Input("dy")
                         .Output("dx")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const bool log_target) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<bool>("log_target", log_target));

    return OpInterpUtil::Dispatch<Tensor>(*op_, {input, target, dy}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class MarginRankingLossFunctor : public LossFunctorBase {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input_1,
                           const std::shared_ptr<one::Tensor>& input_2,
                           const std::shared_ptr<one::Tensor>& target, const float margin,
                           const std::string& reduction) const {
    const auto out =
        sequence_function(functional::Sub)
            .then(functional::Negative)
            .then(std::bind(functional::Mul, target, std::placeholders::_1))
            .then([&margin](const std::shared_ptr<one::Tensor>& x) {
              return functional::ScalarAdd(x, Scalar(margin), /*alpha=*/1, /*inplace=*/true);
            })
            .then(std::bind(functional::Clamp, std::placeholders::_1, Scalar(0), NullOpt))
            .call(input_1, input_2);
    return apply_reduction(out, reduction);
  }
};

class SoftTargetCrossEntropyFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target) const {
    return SequenceFunction<Maybe<Tensor>(const std::shared_ptr<one::Tensor>&)>(
               [](const auto& x) { return functional::LogSoftmax(x, -1); })
        .then([&](const auto& x) { return functional::Mul(x, target); })
        .then(functional::Negative)
        .then([](const auto& x) { return functional::ReduceSum(x, {-1}, false); })
        .then([](const auto& x) { return functional::ReduceMean(x, {}, false); })
        .call(input);
  }
};

class JsdCrossEntropyFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target, const int32_t num_splits,
                           const float alpha) const {
    const int64_t num_batchs = input->shape()->At(0);
    const int64_t split_size = num_batchs / num_splits;
    CHECK_EQ_OR_RETURN(split_size * num_splits, num_batchs)
        << "The batch_size of input should be devisible by num_splits.";
    const auto input_splited = JUST(functional::Split(input, split_size, 0));
    const auto target_splited = JUST(functional::Split(target, split_size, 0));

    const auto loss = functional::CrossEntropy(input_splited->at(0), target_splited->at(0), NullOpt,
                                               -100, "mean");

    TensorTuple probs;
    for (const auto& x : *input_splited) { probs.emplace_back(JUST(functional::Softmax(x, 1))); }

    const auto logp_mixture =
        SequenceFunction<Maybe<Tensor>(const one::TensorTuple&)>(
            [](const auto& x) { return functional::Stack(x, 0); })
            .then([](const auto& x) { return functional::ReduceMean(x, {0}, false); })
            .then([](const auto& x) { return functional::Clamp(x, 1e-7, 1); })
            .then(functional::Log)
            .call(probs);

    TensorTuple kl_div_results;
    for (const auto& x : probs) {
      kl_div_results.emplace_back(JUST(functional::Reshape(
          JUST(functional::KLDivLoss(JUST(logp_mixture), x, false, "batchmean")), {1})));
    }
    const auto result =
        SequenceFunction<Maybe<Tensor>(const one::TensorTuple&)>(
            [](const auto& x) { return functional::Concat(x, 0); })
            .then([](const auto& x) { return functional::ReduceSum(x, {0}, false); })
            .then([&](const auto& x) { return functional::ScalarMul(x, alpha, false); })
            .then([&](const auto& x) { return functional::ScalarDiv(x, probs.size()); })
            .call(kl_div_results);
    return functional::Add(JUST(loss), JUST(result), 1, true);
  }
};

class BinaryCrossEntropyLossFunctor : public LossFunctorBase {
 public:
  BinaryCrossEntropyLossFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy")
                         .Input("input")
                         .Input("target")
                         .Output("out")
                         .Build());
    op_weight_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy")
                                .Input("input")
                                .Input("target")
                                .Input("weight")
                                .Output("out")
                                .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const Optional<one::Tensor>& weight,
                           const std::string& reduction) const {
    MutableAttrMap attrs;
    auto out =
        weight ? OpInterpUtil::Dispatch<Tensor>(*op_weight_, {input, target, JUST(weight)}, attrs)
               : OpInterpUtil::Dispatch<Tensor>(*op_, {input, target}, attrs);
    return apply_reduction(out, reduction);
  }

 private:
  std::shared_ptr<OpExpr> op_;
  std::shared_ptr<OpExpr> op_weight_;
};

class BinaryCrossEntropyLossGradFunctor {
 public:
  BinaryCrossEntropyLossGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy_grad")
                         .Input("input")
                         .Input("target")
                         .Input("dy")
                         .Output("dx")
                         .Build());
    op_weight_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy_grad")
                                .Input("input")
                                .Input("target")
                                .Input("weight")
                                .Input("dy")
                                .Output("dx")
                                .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const Optional<one::Tensor>& weight) const {
    MutableAttrMap attrs;

    if (weight) {
      return OpInterpUtil::Dispatch<one::Tensor>(*op_weight_, {input, target, JUST(weight), dy},
                                                 attrs);
    } else {
      return OpInterpUtil::Dispatch<one::Tensor>(*op_, {input, target, dy}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
  std::shared_ptr<OpExpr> op_weight_;
};

class BinaryCrossEntropyWithLogitsLossFunctor : public LossFunctorBase {
 public:
  BinaryCrossEntropyWithLogitsLossFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy_with_logits")
                         .Input("input")
                         .Input("target")
                         .Output("out")
                         .Build());
    op_weight_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy_with_logits")
                                .Input("input")
                                .Input("target")
                                .Input("weight")
                                .Output("out")
                                .Build());
    op_pos_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy_with_logits")
                             .Input("input")
                             .Input("target")
                             .Input("pos_weight")
                             .Output("out")
                             .Build());
    op_weight_pos_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy_with_logits")
                                    .Input("input")
                                    .Input("target")
                                    .Input("weight")
                                    .Input("pos_weight")
                                    .Output("out")
                                    .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const Optional<one::Tensor>& weight,
                           const Optional<one::Tensor>& pos_weight,
                           const std::string& reduction) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<bool>("has_pos_weight", pos_weight.has_value()));

    std::shared_ptr<Tensor> out;
    if (weight) {
      if (pos_weight) {
        out = JUST(OpInterpUtil::Dispatch<Tensor>(
            *op_weight_pos_, {input, target, JUST(weight), JUST(pos_weight)}, attrs));
      } else {
        out =
            JUST(OpInterpUtil::Dispatch<Tensor>(*op_weight_, {input, target, JUST(weight)}, attrs));
      }
    } else {
      if (pos_weight) {
        out = JUST(
            OpInterpUtil::Dispatch<Tensor>(*op_pos_, {input, target, JUST(pos_weight)}, attrs));
      } else {
        out = JUST(OpInterpUtil::Dispatch<Tensor>(*op_, {input, target}, attrs));
      }
    }
    return apply_reduction(out, reduction);
  }

 private:
  std::shared_ptr<OpExpr> op_;
  std::shared_ptr<OpExpr> op_weight_;
  std::shared_ptr<OpExpr> op_pos_;
  std::shared_ptr<OpExpr> op_weight_pos_;
};

class BinaryCrossEntropyWithLogitsLossGradFunctor {
 public:
  BinaryCrossEntropyWithLogitsLossGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy_with_logits_grad")
                         .Input("input")
                         .Input("target")
                         .Input("dy")
                         .Output("dx")
                         .Build());
    op_weight_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy_with_logits_grad")
                                .Input("input")
                                .Input("target")
                                .Input("weight")
                                .Input("dy")
                                .Output("dx")
                                .Build());
    op_pos_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy_with_logits_grad")
                             .Input("input")
                             .Input("target")
                             .Input("pos_weight")
                             .Input("dy")
                             .Output("dx")
                             .Build());
    op_weight_pos_ = CHECK_JUST(one::OpBuilder("binary_cross_entropy_with_logits_grad")
                                    .Input("input")
                                    .Input("target")
                                    .Input("weight")
                                    .Input("pos_weight")
                                    .Input("dy")
                                    .Output("dx")
                                    .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const Optional<one::Tensor>& weight,
                           const Optional<one::Tensor>& pos_weight) const {
    MutableAttrMap attrs;

    JUST(attrs.SetAttr<bool>("has_pos_weight", pos_weight.has_value()));

    if (weight) {
      if (pos_weight) {
        return OpInterpUtil::Dispatch<one::Tensor>(
            *op_weight_pos_, {input, target, JUST(weight), JUST(pos_weight), dy}, attrs);
      } else {
        return OpInterpUtil::Dispatch<one::Tensor>(*op_weight_, {input, target, JUST(weight), dy},
                                                   attrs);
      }
    } else {
      if (pos_weight) {
        return OpInterpUtil::Dispatch<one::Tensor>(*op_pos_, {input, target, JUST(pos_weight), dy},
                                                   attrs);
      } else {
        return OpInterpUtil::Dispatch<one::Tensor>(*op_, {input, target, dy}, attrs);
      }
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
  std::shared_ptr<OpExpr> op_weight_;
  std::shared_ptr<OpExpr> op_pos_;
  std::shared_ptr<OpExpr> op_weight_pos_;
};

class NllLossFunctor {
 public:
  NllLossFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("nll")
                         .Input("input")
                         .Input("target")
                         .Output("out")
                         .Output("total_weight")
                         .Build());
    op_weight_ = CHECK_JUST(one::OpBuilder("nll")
                                .Input("input")
                                .Input("target")
                                .Input("weight")
                                .Output("out")
                                .Output("total_weight")
                                .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const Optional<one::Tensor>& weight, const int64_t& ignore_index,
                           const std::string& reduction) const {
    CHECK_OR_RETURN(reduction == "none" || reduction == "sum" || reduction == "mean")
        << "Reduction should be none, sum or mean.";

    const auto& input_shape = input->shape();
    const auto& target_shape = target->shape();
    CHECK_LE_OR_RETURN(input_shape->NumAxes(), 5);
    CHECK_EQ_OR_RETURN(input_shape->NumAxes() - 1, target_shape->NumAxes());

    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("ignore_index", ignore_index));

    std::vector<int> input_perm(input_shape->dim_vec().size(), 0);
    input_perm[input_perm.size() - 1] = 1;
    for (size_t i = 1; i < input_perm.size() - 1; ++i) { input_perm[i] = i + 1; }

    const auto input_ = JUST(sequence_function(functional::Transpose)
                                 .then(std::bind(functional::Reshape, std::placeholders::_1,
                                                 Shape({-1, input_shape->At(1)})))
                                 .call(input, input_perm));
    auto target_ = JUST(functional::Flatten(target, 0, target_shape->NumAxes() - 1));

    std::shared_ptr<TensorTuple> kernel_result;
    std::shared_ptr<Tensor> result;
    if (weight) {
      kernel_result = JUST(
          OpInterpUtil::Dispatch<TensorTuple>(*op_weight_, {input_, target_, JUST(weight)}, attrs));
    } else {
      kernel_result = JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_, {input_, target_}, attrs));
    }
    result = JUST(functional::Reshape(kernel_result->at(0), *target_shape));
    if (reduction == "none") { return result; }

    result = JUST(functional::ReduceSum(result, {}, false));

    if (reduction == "sum") { return result; }

    return functional::Div(result, kernel_result->at(1));
  }

 private:
  std::shared_ptr<OpExpr> op_;
  std::shared_ptr<OpExpr> op_weight_;
};

class NllLossGradFunctor {
 public:
  NllLossGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("nll_grad")
                         .Input("input")
                         .Input("target")
                         .Input("total_weight")
                         .Input("dy")
                         .Output("dx")
                         .Build());
    op_weight_ = CHECK_JUST(one::OpBuilder("nll_grad")
                                .Input("input")
                                .Input("target")
                                .Input("total_weight")
                                .Input("weight")
                                .Input("dy")
                                .Output("dx")
                                .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const Optional<one::Tensor>& weight,
                           const std::shared_ptr<one::Tensor>& total_weight,
                           const int64_t ignore_index) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("ignore_index", ignore_index));

    if (weight) {
      return OpInterpUtil::Dispatch<one::Tensor>(
          *op_weight_, {input, target, total_weight, JUST(weight), dy}, attrs);
    } else {
      return OpInterpUtil::Dispatch<one::Tensor>(*op_, {input, target, total_weight, dy}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
  std::shared_ptr<OpExpr> op_weight_;
};

class CrossEntropyFunctor {
 public:
  CrossEntropyFunctor() {
    op_log_softmax_ = CHECK_JUST(one::OpBuilder("log_softmax").Input("in").Output("prob").Build());
    op_nll_ = CHECK_JUST(one::OpBuilder("nll")
                             .Input("input")
                             .Input("target")
                             .Output("out")
                             .Output("total_weight")
                             .Build());
    op_nll_weight_ = CHECK_JUST(one::OpBuilder("nll")
                                    .Input("input")
                                    .Input("target")
                                    .Input("weight")
                                    .Output("out")
                                    .Output("total_weight")
                                    .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& target,
                           const Optional<one::Tensor>& weight, const int64_t& ignore_index,
                           const std::string& reduction) const {
    CHECK_OR_RETURN(reduction == "none" || reduction == "sum" || reduction == "mean")
        << "Reduction should be none, sum or mean.";
    const auto& input_shape = input->shape();
    const auto& target_shape = target->shape();
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("ignore_index", ignore_index));

    std::vector<int> input_perm(input_shape->dim_vec().size(), 0);
    input_perm[input_perm.size() - 1] = 1;
    for (size_t i = 1; i < input_perm.size() - 1; ++i) { input_perm[i] = i + 1; }

    const auto input_ = JUST(sequence_function(functional::Transpose)
                                 .then(std::bind(functional::Reshape, std::placeholders::_1,
                                                 Shape({-1, input_shape->At(1)})))
                                 .then([this](const std::shared_ptr<one::Tensor>& x) {
                                   return OpInterpUtil::Dispatch<Tensor>(*op_log_softmax_, {x});
                                 })
                                 .call(input, input_perm));

    const auto target_ = JUST(functional::Flatten(target, 0, target->shape()->NumAxes() - 1));

    std::shared_ptr<TensorTuple> kernel_result;
    std::shared_ptr<Tensor> result;
    if (weight) {
      kernel_result = JUST(OpInterpUtil::Dispatch<TensorTuple>(
          *op_nll_weight_, {input_, target_, JUST(weight)}, attrs));
    } else {
      kernel_result = JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_nll_, {input_, target_}, attrs));
    }
    result = JUST(functional::Reshape(kernel_result->at(0), *target_shape));
    if (reduction == "none") { return result; }

    result = JUST(functional::ReduceSum(result, {}, false));
    if (reduction == "sum") { return result; }

    return functional::Div(result, kernel_result->at(1));
  }

 private:
  std::shared_ptr<OpExpr> op_log_softmax_;
  std::shared_ptr<OpExpr> op_nll_;
  std::shared_ptr<OpExpr> op_nll_weight_;
};

class SparseCrossEntropyFunctor {
 public:
  SparseCrossEntropyFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("sparse_cross_entropy")
                         .Input("prediction")
                         .Input("label")
                         .Output("out")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& prediction,
                           const std::shared_ptr<one::Tensor>& label, const int64_t& depth) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("depth", depth));

    return OpInterpUtil::Dispatch<Tensor>(*op_, {prediction, label}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class SparseCrossEntropyGradFunctor {
 public:
  SparseCrossEntropyGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("sparse_cross_entropy_grad")
                         .Input("prediction")
                         .Input("label")
                         .Input("dy")
                         .Output("prediction_diff")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& prediction,
                           const std::shared_ptr<one::Tensor>& label,
                           const std::shared_ptr<one::Tensor>& dy, const int64_t& depth) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("depth", depth));

    return OpInterpUtil::Dispatch<Tensor>(*op_, {prediction, label, dy}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class SparseCrossEntropyMsFunctor {
 public:
  SparseCrossEntropyMsFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("sparse_cross_entropy_ms")
                         .Input("prediction")
                         .Input("label")
                         .Output("out")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& prediction,
                           const std::shared_ptr<one::Tensor>& label, const int64_t& depth) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("depth", depth));

    return OpInterpUtil::Dispatch<Tensor>(*op_, {prediction, label}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class SparseCrossEntropyMsGradFunctor {
 public:
  SparseCrossEntropyMsGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("sparse_cross_entropy_ms_grad")
                         .Input("prediction")
                         .Input("label")
                         .Input("dy")
                         .Output("prediction_diff")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& prediction,
                           const std::shared_ptr<one::Tensor>& label,
                           const std::shared_ptr<one::Tensor>& dy, const int64_t& depth) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("depth", depth));

    return OpInterpUtil::Dispatch<Tensor>(*op_, {prediction, label, dy}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class SparseSoftmaxCrossEntropyFunctor {
 public:
  SparseSoftmaxCrossEntropyFunctor() {
    // SparseSoftmaxCrossEntropy
    op_sparse_softmax_cross_entropy_ = CHECK_JUST(one::OpBuilder("sparse_softmax_cross_entropy")
                                                      .Input("prediction")
                                                      .Input("label")
                                                      .Output("prob")
                                                      .Output("out")
                                                      .Build());
    // lazy model SparseSoftmaxCrossEntropyMs
    op_sparse_softmax_cross_entropy_ms_ =
        CHECK_JUST(one::OpBuilder("sparse_softmax_cross_entropy_ms")
                       .Input("prediction")
                       .Input("label")
                       .Output("prob")
                       .Output("out")
                       .Build());
    // eager model SparseSoftmaxCrossEntropyMs
    op_reduce_max_device_stage_ = CHECK_JUST(one::OpBuilder("reduce_max_device_stage")
                                                 .Input("in")
                                                 .Output("out")
                                                 .Output("mask")
                                                 .Output("count")
                                                 .Build());
    op_reduce_max_global_stage_ = CHECK_JUST(one::OpBuilder("reduce_max_global_stage")
                                                 .Input("in")
                                                 .Input("device_count")
                                                 .Output("out")
                                                 .Output("mask")
                                                 .Build());
    op_sparse_cross_entropy_ms_ = CHECK_JUST(one::OpBuilder("sparse_cross_entropy_ms")
                                                 .Input("prediction")
                                                 .Input("label")
                                                 .Output("out")
                                                 .Build());
    op_broadcast_sub_ =
        CHECK_JUST(one::OpBuilder("broadcast_sub").Input("x").Input("y").Output("z").Build());
    op_broadcast_div_ =
        CHECK_JUST(one::OpBuilder("broadcast_div").Input("x").Input("y").Output("z").Build());
    op_reduce_sum_ = CHECK_JUST(
        one::OpBuilder("reduce_sum").Input("input_tensor").Output("output_tensor").Build());
    op_exp_ = CHECK_JUST(one::OpBuilder("exp").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& logits,
                           const std::shared_ptr<one::Tensor>& label) const {
    if (JUST(RunWithMsVersion(logits, label))) {
      if (LazyMode::is_enabled()) {
        return LazySparseSoftmaxCrossEntropyMsOperator(logits, label);
      } else {
        return EagerSparseSoftmaxCrossEntropyMsOperator(logits, label);
      }
    } else {
      return SparseSoftmaxCrossEntropyOperator(logits, label);
    }
  }

  Maybe<bool> RunWithMsVersion(const std::shared_ptr<one::Tensor>& logits,
                               const std::shared_ptr<one::Tensor>& label) const {
    if (!(logits->is_consistent() && label->is_consistent())) { return false; }

    if (logits->shape()->NumAxes() != 2) { return false; }

    const cfg::NdSbp& logits_nd_sbp = *(JUST(logits->nd_sbp()));
    const int32_t split_axis = logits->shape()->NumAxes() - 1;
    bool has_split_axis_parallel = false;
    for (int64_t i = 0; i < logits_nd_sbp.sbp_parallel_size(); ++i) {
      const auto& sbp = logits_nd_sbp.sbp_parallel(i);
      if (sbp.has_split_parallel() && sbp.split_parallel().axis() == split_axis) {
        has_split_axis_parallel = true;
      } else {
        if (sbp.has_partial_sum_parallel()) { return false; }
      }
    }
    if (!has_split_axis_parallel) { return false; }

    return true;
  }

  Maybe<Tensor> SparseSoftmaxCrossEntropyOperator(const std::shared_ptr<one::Tensor>& logits,
                                                  const std::shared_ptr<one::Tensor>& label) const {
    MutableAttrMap attrs;
    int64_t depth = logits->shape()->At(logits->shape()->NumAxes() - 1);
    JUST(attrs.SetAttr<int64_t>("depth", depth));
    const auto& result = JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_sparse_softmax_cross_entropy_,
                                                                  {logits, label}, attrs));
    return result->at(1);
  }

  Maybe<Tensor> LazySparseSoftmaxCrossEntropyMsOperator(
      const std::shared_ptr<one::Tensor>& logits, const std::shared_ptr<one::Tensor>& label) const {
    MutableAttrMap attrs;
    int64_t depth = logits->shape()->At(logits->shape()->NumAxes() - 1);
    JUST(attrs.SetAttr<int64_t>("depth", depth));
    const auto& result = JUST(OpInterpUtil::Dispatch<TensorTuple>(
        *op_sparse_softmax_cross_entropy_ms_, {logits, label}, attrs));
    return result->at(1);
  }

  Maybe<Tensor> EagerSparseSoftmaxCrossEntropyMsOperator(
      const std::shared_ptr<one::Tensor>& logits, const std::shared_ptr<one::Tensor>& label) const {
    // op_reduce_max_device_stage_
    MutableAttrMap attrs;
    int64_t depth = logits->shape()->At(logits->shape()->NumAxes() - 1);
    int32_t axis = logits->shape()->NumAxes() - 1;
    JUST(attrs.SetAttr<std::vector<int32_t>>("axis", {axis}));
    const auto& max_device_stage =
        JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_reduce_max_device_stage_, {logits}, attrs));
    std::shared_ptr<Tensor> max_global_stage_input0 = max_device_stage->at(0);
    std::shared_ptr<Tensor> max_global_stage_input1 = max_device_stage->at(2);

    const cfg::NdSbp& logits_nd_sbp = *(JUST(logits->nd_sbp()));
    std::vector<Symbol<cfg::SbpParallel>> s0b_sbp_parallels;
    std::vector<Symbol<cfg::SbpParallel>> s0s1_sbp_parallels;
    if (logits_nd_sbp.sbp_parallel_size() == 2) {
      cfg::SbpParallel sbp;
      sbp.mutable_broadcast_parallel();
      s0b_sbp_parallels.emplace_back(logits_nd_sbp.sbp_parallel(0));
      s0b_sbp_parallels.emplace_back(sbp);
      s0s1_sbp_parallels.emplace_back(logits_nd_sbp.sbp_parallel(0));
      s0s1_sbp_parallels.emplace_back(logits_nd_sbp.sbp_parallel(1));
      max_global_stage_input0 = JUST(functional::ToConsistent(
          max_device_stage->at(0), JUST(max_device_stage->at(0)->parallel_desc()),
          s0b_sbp_parallels, s0s1_sbp_parallels));
      max_global_stage_input1 = JUST(functional::ToConsistent(
          max_device_stage->at(2), JUST(max_device_stage->at(0)->parallel_desc()),
          s0b_sbp_parallels, s0s1_sbp_parallels));
    }
    // op_reduce_max_global_stage_
    attrs.clear();
    JUST(attrs.SetAttr<std::vector<int32_t>>("axis", {axis}));
    JUST(attrs.SetAttr<bool>("keepdims", true));
    const auto& max_global_stage = JUST(OpInterpUtil::Dispatch<TensorTuple>(
        *op_reduce_max_global_stage_, {max_global_stage_input0, max_global_stage_input1}, attrs));
    auto& broadcast_sub_input = max_global_stage->at(0);
    if (logits_nd_sbp.sbp_parallel_size() == 2) {
      broadcast_sub_input = JUST(functional::ToConsistent(
          broadcast_sub_input, JUST(max_device_stage->at(0)->parallel_desc()), s0b_sbp_parallels,
          s0b_sbp_parallels));
    }
    // op_broadcast_sub_
    attrs.clear();
    const auto& output_broadcast_sub = JUST(OpInterpUtil::Dispatch<TensorTuple>(
        *op_broadcast_sub_, {logits, broadcast_sub_input}, attrs));
    // op_exp_
    const auto& output_exp =
        JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_exp_, {output_broadcast_sub->at(0)}, attrs));
    // op_reduce_sum_
    JUST(attrs.SetAttr<std::vector<int32_t>>("axis", {axis}));
    JUST(attrs.SetAttr<bool>("keepdims", true));
    const auto& output_reduce_sum =
        JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_reduce_sum_, {output_exp->at(0)}, attrs));
    std::shared_ptr<Tensor> broadcast_div_input1 = output_reduce_sum->at(0);
    if (logits_nd_sbp.sbp_parallel_size() == 2) {
      std::vector<Symbol<cfg::SbpParallel>> empty_grad_sbp_parallels;
      broadcast_div_input1 = JUST(functional::ToConsistent(
          output_reduce_sum->at(0), JUST(output_reduce_sum->at(0)->parallel_desc()),
          s0b_sbp_parallels, s0b_sbp_parallels));
    }
    // op_broadcast_div_
    attrs.clear();
    const auto& predictions = JUST(OpInterpUtil::Dispatch<TensorTuple>(
        *op_broadcast_div_, {output_exp->at(0), broadcast_div_input1}, attrs));
    // op_sparse_cross_entropy_ms_
    JUST(attrs.SetAttr<int64_t>("depth", depth));
    const auto& output = JUST(OpInterpUtil::Dispatch<Tensor>(*op_sparse_cross_entropy_ms_,
                                                             {predictions->at(0), label}, attrs));
    return output;
  }

 private:
  // SparseSoftmaxCrossEntropy
  std::shared_ptr<OpExpr> op_sparse_softmax_cross_entropy_;
  // lazy model SparseSoftmaxCrossEntropyMs
  std::shared_ptr<OpExpr> op_sparse_softmax_cross_entropy_ms_;
  // SparseSoftmaxCrossEntropyMs
  std::shared_ptr<OpExpr> op_reduce_max_device_stage_;
  std::shared_ptr<OpExpr> op_reduce_max_global_stage_;
  std::shared_ptr<OpExpr> op_broadcast_sub_;
  std::shared_ptr<OpExpr> op_exp_;
  std::shared_ptr<OpExpr> op_reduce_sum_;
  std::shared_ptr<OpExpr> op_broadcast_div_;
  std::shared_ptr<OpExpr> op_sparse_cross_entropy_ms_;
};

class SparseSoftmaxCrossEntropyGrad {
 public:
  SparseSoftmaxCrossEntropyGrad() {
    op_ = CHECK_JUST(one::OpBuilder("sparse_softmax_cross_entropy_grad")
                         .Input("prob")
                         .Input("label")
                         .Input("dy")
                         .Output("prediction_diff")
                         .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& prob,
                           const std::shared_ptr<one::Tensor>& label, const int64_t& depth) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("depth", depth));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {prob, label, dy}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class SoftmaxCrossEntropyFunctor {
 public:
  SoftmaxCrossEntropyFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("softmax_cross_entropy")
                         .Input("prediction")
                         .Input("label")
                         .Output("out")
                         .Output("prob")
                         .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& logits,
                           const std::shared_ptr<one::Tensor>& label) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {logits, label});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class SoftmaxCrossEntropyGradFunctor {
 public:
  SoftmaxCrossEntropyGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("softmax_cross_entropy_grad")
                         .Input("dy")
                         .Input("label")
                         .Input("prob")
                         .Output("prediction_diff")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& label,
                           const std::shared_ptr<one::Tensor>& prob) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, label, prob});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class CombinedMarginLossFunctor {
 public:
  CombinedMarginLossFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("combined_margin_loss")
                         .Input("x")
                         .Input("label")
                         .Output("y")
                         .Output("theta")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& label, const float& m1,
                           const float& m2, const float& m3) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("m1", m1));
    JUST(attrs.SetAttr<float>("m2", m2));
    JUST(attrs.SetAttr<float>("m3", m3));
    JUST(attrs.SetAttr<int64_t>("depth", x->shape()->At(1)));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, label}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class CombinedMarginLossGradFunctor {
 public:
  CombinedMarginLossGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("combined_margin_loss_grad")
                         .Input("dy")
                         .Input("label")
                         .Input("theta")
                         .Output("dx")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& label,
                           const std::shared_ptr<one::Tensor>& theta, const float& m1,
                           const float& m2, const float& m3, const int64_t& depth) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("m1", m1));
    JUST(attrs.SetAttr<float>("m2", m2));
    JUST(attrs.SetAttr<float>("m3", m3));
    JUST(attrs.SetAttr<int64_t>("depth", depth));
    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {dy, label, theta}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class CtcLossFunctor {
 public:
  CtcLossFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("ctc_loss")
                         .Input("log_probs")
                         .Input("targets")
                         .Input("input_lengths")
                         .Input("target_lengths")
                         .Output("loss")
                         .Output("alpha")
                         .Build());
    op_xdivy_ = CHECK_JUST(one::OpBuilder("xdivy").Input("x").Input("y").Output("z").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& log_probs,
                           const std::shared_ptr<one::Tensor>& targets,
                           const std::shared_ptr<one::Tensor>& input_lengths,
                           const std::shared_ptr<one::Tensor>& target_lengths,
                           const int64_t& max_target_length, const int& blank,
                           const bool& zero_infinity, const std::string& reduction) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("max_target_length", max_target_length));
    JUST(attrs.SetAttr<int32_t>("blank", blank));
    JUST(attrs.SetAttr<bool>("zero_infinity", zero_infinity));
    auto out = JUST(OpInterpUtil::Dispatch<Tensor>(
        *op_, {log_probs, targets, input_lengths, target_lengths}, attrs));
    if (zero_infinity) {
      const auto create_constant = [&](const Scalar& scalar) -> Maybe<Tensor> {
        return functional::Constant(*out->shape(), scalar, out->dtype(), JUST(out->device()));
      };

      out = JUST(sequence_function(functional::Constant)
                     .then(std::bind(functional::BroadcastEqual, out, std::placeholders::_1))
                     .then(std::bind(functional::Where, std::placeholders::_1,
                                     JUST(create_constant(Scalar(0))), out))
                     .call(*out->shape(), Scalar(std::numeric_limits<double>::infinity()),
                           out->dtype(), JUST(out->device())));
    }
    CHECK_OR_RETURN([&]() -> bool {
      if ((reduction != "none") && (reduction != "sum") && (reduction != "mean")) return false;
      return true;
    }());
    if (reduction == "sum") { return functional::ReduceSum(out, {}, false); }
    if (reduction == "mean") {
      return sequence_function(functional::Clamp)
          .then(std::bind(functional::Cast, std::placeholders::_1, log_probs->dtype()))
          .then([&](const std::shared_ptr<one::Tensor>& x) {
            return OpInterpUtil::Dispatch<Tensor>(*op_xdivy_, {out, x});
          })
          .then(std::bind(functional::ReduceMean, std::placeholders::_1, std::vector<int32_t>({}),
                          false))
          .call(target_lengths, Scalar(1), NullOpt);
    }
    return out;
  }

 private:
  std::shared_ptr<OpExpr> op_;
  std::shared_ptr<OpExpr> op_xdivy_;
};

class CtcLossGradFunctor {
 public:
  CtcLossGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("ctc_loss_grad")
                         .Input("grad_out")
                         .Input("log_probs")
                         .Input("targets")
                         .Input("input_lengths")
                         .Input("target_lengths")
                         .Input("loss")
                         .Input("alpha")
                         .Output("grad")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& grad_out,
                           const std::shared_ptr<one::Tensor>& log_probs,
                           const std::shared_ptr<one::Tensor>& targets,
                           const std::shared_ptr<one::Tensor>& input_lengths,
                           const std::shared_ptr<one::Tensor>& target_lengths,
                           const std::shared_ptr<one::Tensor>& loss,
                           const std::shared_ptr<one::Tensor>& alpha, const int32_t& blank,
                           const bool& zero_infinity, const int64_t& max_target_length) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("blank", blank));
    JUST(attrs.SetAttr<bool>("zero_infinity", zero_infinity));
    JUST(attrs.SetAttr<int64_t>("max_target_length", max_target_length));
    return OpInterpUtil::Dispatch<one::Tensor>(
        *op_, {grad_out, log_probs, targets, input_lengths, target_lengths, loss, alpha}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class TripletMarginLossFunctor {
 public:
  TripletMarginLossFunctor() {}

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& anchor,
                           const std::shared_ptr<one::Tensor>& positive,
                           const std::shared_ptr<one::Tensor>& negative, const float& margin,
                           const float& p, const float& eps, const bool& swap,
                           const std::string& reduction) const {
    int32_t dim_norm = anchor->ndim() - 1;
    std::vector<int32_t> dim(1, dim_norm);
    CHECK_OR_RETURN([&]() -> bool {
      if ((reduction != "none") && (reduction != "sum") && (reduction != "mean")) return false;
      return true;
    }());
    auto da_p = JUST(VectorNorm(JUST(ScalarAdd(eps, JUST(Sub(anchor, positive)), /*alpha=*/1)), p,
                                dim, /*keepdim=*/false, anchor->dtype()));
    auto da_n = JUST(VectorNorm(JUST(ScalarAdd(eps, JUST(Sub(anchor, negative)), /*alpha=*/1)), p,
                                dim, /*keepdim=*/false, anchor->dtype()));
    if (swap) {
      auto distance_swap =
          JUST(VectorNorm(JUST(ScalarAdd(eps, JUST(Sub(positive, negative)), /*alpha=*/1)), p, dim,
                          /*keepdim=*/false, positive->dtype()));
      da_n = JUST(Minimum(distance_swap, da_n));
    }
    auto triplet_loss =
        JUST(Clamp(JUST(ScalarAdd(JUST(Sub(da_p, da_n)), margin, /*alpha=*/1, /*inplace=*/false)),
                   /*min=*/0.0, NullOpt));
    int32_t ndim = triplet_loss->ndim() - 1;
    std::vector<int32_t> axis(1, ndim);

    if (reduction == "mean") {
      triplet_loss = JUST(ReduceMean(triplet_loss, axis, /*keepdim=*/false));
    } else if (reduction == "sum") {
      triplet_loss = JUST(ReduceSum(triplet_loss, axis, /*keepdim=*/false));
    }
    return triplet_loss;
  }
};
}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  // functors of the losses which have no kernel
  m.add_functor<impl::L1LossFunctor>("L1Loss");
  m.add_functor<impl::MseLossFunctor>("MseLoss");
  m.add_functor<impl::CrossEntropyFunctor>("CrossEntropy");
  m.add_functor<impl::SoftTargetCrossEntropyFunctor>("SoftTargetCrossEntropy");
  m.add_functor<impl::TripletMarginLossFunctor>("TripletMarginLoss");
  m.add_functor<impl::MarginRankingLossFunctor>("MarginRankingLoss");
  m.add_functor<impl::JsdCrossEntropyFunctor>("JsdCrossEntropy");

  // functors of the losses which have kernels
  m.add_functor<impl::KLDivLossFunctor>("KLDivLoss");
  m.add_functor<impl::NllLossFunctor>("NllLoss");
  m.add_functor<impl::BinaryCrossEntropyLossFunctor>("BinaryCrossEntropyLoss");
  m.add_functor<impl::BinaryCrossEntropyWithLogitsLossFunctor>("BinaryCrossEntropyWithLogitsLoss");
  m.add_functor<impl::SparseCrossEntropyFunctor>("SparseCrossEntropy");
  m.add_functor<impl::SparseCrossEntropyMsFunctor>("SparseCrossEntropyMs");
  m.add_functor<impl::SparseSoftmaxCrossEntropyFunctor>("SparseSoftmaxCrossEntropy");
  m.add_functor<impl::SoftmaxCrossEntropyFunctor>("SoftmaxCrossEntropy");
  m.add_functor<impl::SmoothL1LossFunctor>("SmoothL1Loss");
  m.add_functor<impl::CombinedMarginLossFunctor>("CombinedMarginLoss");
  m.add_functor<impl::CtcLossFunctor>("CtcLoss");

  // grad functors of the losses which have kernels
  m.add_functor<impl::KLDivLossInputGradFunctor>("KLDivLossInputGrad");
  m.add_functor<impl::KLDivLossTargetGradFunctor>("KLDivLossTargetGrad");
  m.add_functor<impl::NllLossGradFunctor>("NllLossGrad");
  m.add_functor<impl::BinaryCrossEntropyLossGradFunctor>("BinaryCrossEntropyLossGrad");
  m.add_functor<impl::BinaryCrossEntropyWithLogitsLossGradFunctor>(
      "BinaryCrossEntropyWithLogitsLossGrad");
  m.add_functor<impl::SparseCrossEntropyGradFunctor>("SparseCrossEntropyGrad");
  m.add_functor<impl::SparseCrossEntropyMsGradFunctor>("SparseCrossEntropyMsGrad");
  m.add_functor<impl::SparseSoftmaxCrossEntropyGrad>("SparseSoftmaxCrossEntropyGrad");
  m.add_functor<impl::SoftmaxCrossEntropyGradFunctor>("SoftmaxCrossEntropyGrad");

  m.add_functor<impl::SmoothL1LossGradFunctor>("SmoothL1LossGrad");
  m.add_functor<impl::CombinedMarginLossGradFunctor>("CombinedMarginLossGrad");
  m.add_functor<impl::CtcLossGradFunctor>("CtcLossGrad");
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow