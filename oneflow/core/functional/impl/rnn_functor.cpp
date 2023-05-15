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

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/error.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/tensor_util.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/random_generator.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/sequence_function.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/functional/impl/unary_functor.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/user/kernels/distributions/common.h"
#include "oneflow/core/framework/nd_sbp.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {
// NOTE(Liang Depeng): The implementation of rnn related functors are modified from
//                     https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/RNN.cpp
struct tanh_f {
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& t) const {
    return JUST(functional::Tanh(t));
  }
};
struct relu_f {
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& t) const {
    return JUST(functional::Relu(t, false));
  }
};

Maybe<void> check_rnn_cell_forward_input(const std::shared_ptr<one::Tensor>& input,
                                         int64_t input_size) {
  CHECK_OR_RETURN(input->shape()->At(1) == input_size)
      << "input has inconsistent input_size: got " << input->shape()->At(1) << " expected "
      << input_size;
  return Maybe<void>::Ok();
}

Maybe<void> check_rnn_cell_forward_hidden(const std::shared_ptr<one::Tensor>& input,
                                          const std::shared_ptr<one::Tensor>& hx,
                                          int64_t hidden_size, int64_t hidden_label) {
  CHECK_OR_RETURN(input->shape()->At(0) == hx->shape()->At(0))
      << "Input batch size " << input->shape()->At(0) << " doesn't match hidden" << hidden_label
      << " batch size " << hx->shape()->At(0);

  CHECK_OR_RETURN(hx->shape()->At(1) == hidden_size)
      << "hidden" << hidden_label << " has inconsistent hidden_size: got " << hx->shape()->At(1)
      << ", expected " << hidden_size;
  return Maybe<void>::Ok();
}

Maybe<void> check_attributes(const std::shared_ptr<one::Tensor>& input, const TensorTuple& params,
                             const TensorTuple& hiddens, bool check_dtype = false) {
  DeviceType input_device{};
  if (input->is_global()) {
    input_device = JUST(input->parallel_desc())->device_type();
  } else {
    input_device = JUST(input->device())->enum_type();
  }

  DataType input_dtype = input->dtype()->data_type();

  auto check_tensors = [&](const std::string& name,
                           const std::shared_ptr<one::Tensor>& t) -> Maybe<void> {
    DeviceType t_device{};
    if (t->is_global()) {
      t_device = JUST(t->parallel_desc())->device_type();
    } else {
      t_device = JUST(t->device())->enum_type();
    }

    CHECK_OR_RETURN(input_device == t_device)
        << "Input and " << name << " tensors are not at the same device, found input tensor at "
        << input_device << " and " << name << " tensor at " << t_device;

    if (check_dtype) {
      DataType t_dtype = t->dtype()->data_type();
      CHECK_OR_RETURN(input_dtype == t_dtype)
          << "Input and " << name << " tensors are not the same dtype, found input tensor with "
          << input_dtype << " and " << name << " tensor with " << t_dtype;
    }
    return Maybe<void>::Ok();
  };

  for (const auto& h : hiddens) JUST(check_tensors("hidden", h));
  for (const auto& p : params) JUST(check_tensors("parameter", p));

  return Maybe<void>::Ok();
}

Maybe<Tensor> linear(const std::shared_ptr<one::Tensor>& input,
                     const std::shared_ptr<one::Tensor>& weight,
                     const std::shared_ptr<one::Tensor>& bias) {
  if (bias != nullptr) {
    TensorTuple weights;
    weights.emplace_back(weight);
    TensorTuple biases;
    biases.emplace_back(bias);
    return functional::FusedMLP(input, weights, biases, true);
  } else {
    return functional::MatMul(input, weight, false, true, 1.0);
  }
}

struct CellParams {
  CellParams(const std::shared_ptr<one::Tensor> _w_ih,  // NOLINT
             const std::shared_ptr<one::Tensor> _w_hh,  // NOLINT
             const std::shared_ptr<one::Tensor> _b_ih,  // NOLINT
             const std::shared_ptr<one::Tensor> _b_hh,  // NOLINT
             const std::shared_ptr<one::Tensor> _w_hr)  // NOLINT
      : w_ih(_w_ih), w_hh(_w_hh), b_ih_(_b_ih), b_hh_(_b_hh), w_hr(_w_hr){};

  const std::shared_ptr<one::Tensor> w_ih;
  const std::shared_ptr<one::Tensor> w_hh;
  const std::shared_ptr<one::Tensor> b_ih_;
  const std::shared_ptr<one::Tensor> b_hh_;
  const std::shared_ptr<one::Tensor> w_hr;  // only defined for LSTMs with projections

  Maybe<Tensor> matmul_ih(const std::shared_ptr<one::Tensor>& input) const {
    return functional::MatMul(input, w_ih, false, true, 1.0);
  }

  Maybe<Tensor> matmul_hh(const std::shared_ptr<one::Tensor>& h) const {
    return functional::MatMul(h, w_hh, false, true, 1.0);
  }

  Maybe<Tensor> matmul_hr(const std::shared_ptr<one::Tensor>& h) const {
    if (w_hr != nullptr) { return functional::MatMul(h, w_hr, false, true, 1.0); }
    return h;
  }

  Maybe<Tensor> linear_ih(const std::shared_ptr<one::Tensor>& input) const {
    return linear(input, w_ih, b_ih_);
  }

  Maybe<Tensor> linear_hh(const std::shared_ptr<one::Tensor>& h) const {
    return linear(h, w_hh, b_hh_);
  }

  const std::shared_ptr<one::Tensor>& b_ih() const { return b_ih_; }
  const std::shared_ptr<one::Tensor>& b_hh() const { return b_hh_; }
};

// Parses a flat list of parameter tensors into a list of CellParams
static Maybe<std::vector<CellParams>> gather_params(const TensorTuple& params, bool has_biases,
                                                    bool has_projections = false) {
  std::vector<CellParams> result;
  if (has_biases) {
    if (has_projections) {
      CHECK_OR_RETURN(params.size() % 5 == 0) << "got an incorrect number of RNN parameters";
      for (size_t i = 0; i < params.size(); i += 5) {
        result.emplace_back(params[i], params[i + 1], params[i + 2], params[i + 3], params[i + 4]);
      }
    } else {
      CHECK_OR_RETURN(params.size() % 4 == 0) << "got an incorrect number of RNN parameters";
      for (size_t i = 0; i < params.size(); i += 4) {
        result.emplace_back(params[i], params[i + 1], params[i + 2], params[i + 3], nullptr);
      }
    }
  } else {
    if (has_projections) {
      CHECK_OR_RETURN(params.size() % 3 == 0) << "got an incorrect number of RNN parameters";
      for (size_t i = 0; i < params.size(); i += 3) {
        result.emplace_back(params[i], params[i + 1], nullptr, nullptr, params[i + 2]);
      }
    } else {
      CHECK_OR_RETURN(params.size() % 2 == 0) << "got an incorrect number of RNN parameters";
      for (size_t i = 0; i < params.size(); i += 2) {
        result.emplace_back(params[i], params[i + 1], nullptr, nullptr, nullptr);
      }
    }
  }
  return result;
}

template<typename nonlinearity, typename cell_params>
struct SimpleCell {
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& hidden, const cell_params& params,
                           bool pre_compute_input = false) const {
    std::shared_ptr<one::Tensor> hh = JUST(params.linear_hh(hidden));
    std::shared_ptr<one::Tensor> output;
    if (pre_compute_input) {
      output = JUST(functional::Add(hh, input, 1.0, true));
    } else {
      std::shared_ptr<one::Tensor> ih = JUST(params.linear_ih(input));
      output = JUST(functional::Add(hh, ih, 1.0, true));
    }
    return nonlinearity{}(output);
  }
};

template<typename cell_params>
struct GRUCell {
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& hidden, const cell_params& params,
                           bool pre_compute_input = false) const {
    DeviceType input_device{};
    if (input->is_global()) {
      input_device = JUST(input->parallel_desc())->device_type();
    } else {
      input_device = JUST(input->device())->enum_type();
    }

    if (input_device == DeviceType::kCUDA) {
      CHECK_OR_RETURN(!pre_compute_input);

      std::shared_ptr<one::Tensor> igates = JUST(params.matmul_ih(input));
      std::shared_ptr<one::Tensor> hgates = JUST(params.matmul_hh(hidden));

      std::shared_ptr<TensorTuple> result =
          JUST(functional::FusedGruCell(igates, hgates, hidden, params.b_ih(), params.b_hh()));

      return (*result)[0];
    }

    std::shared_ptr<one::TensorTuple> chunked_igates;
    if (pre_compute_input) {
      chunked_igates = JUST(functional::Chunk(input, 3, 1));
    } else {
      std::shared_ptr<one::Tensor> gates_ih = JUST(params.linear_ih(input));
      chunked_igates = JUST(functional::Chunk(gates_ih, 3, 1));
    }

    std::shared_ptr<one::Tensor> tmp = JUST(params.linear_hh(hidden));
    std::shared_ptr<one::TensorTuple> chunked_hgates = JUST(functional::Chunk(tmp, 3, 1));
    std::shared_ptr<one::Tensor> reset_gate =
        JUST(functional::Add((*chunked_hgates)[0], (*chunked_igates)[0], 1.0, false));
    reset_gate = JUST(functional::Sigmoid(reset_gate));
    std::shared_ptr<one::Tensor> input_gate =
        JUST(functional::Add((*chunked_hgates)[1], (*chunked_igates)[1], 1.0, false));
    input_gate = JUST(functional::Sigmoid(input_gate));
    std::shared_ptr<one::Tensor> new_gate = JUST(functional::Mul((*chunked_hgates)[2], reset_gate));
    new_gate = JUST(functional::Add((*chunked_igates)[2], new_gate, 1.0, false));
    new_gate = JUST(functional::Tanh(new_gate));
    std::shared_ptr<one::Tensor> output = JUST(functional::Sub(hidden, new_gate, 1.0, false));
    output = JUST(functional::Mul(output, input_gate));
    output = JUST(functional::Add(output, new_gate, 1.0, false));
    return output;
  }
};

template<typename cell_params>
struct LSTMCell {
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& input,
                                const one::TensorTuple& hidden, const cell_params& params,
                                bool pre_compute_input = false) const {
    const std::shared_ptr<Tensor>& hx = hidden[0];
    const std::shared_ptr<Tensor>& cx = hidden[1];

    DeviceType input_device{};
    if (input->is_global()) {
      input_device = JUST(input->parallel_desc())->device_type();
    } else {
      input_device = JUST(input->device())->enum_type();
    }

    if (input_device == DeviceType::kCUDA) {
      CHECK_OR_RETURN(!pre_compute_input);

      std::shared_ptr<one::Tensor> igates = JUST(params.matmul_ih(input));
      std::shared_ptr<one::Tensor> hgates = JUST(params.matmul_hh(hx));

      std::shared_ptr<TensorTuple> result =
          JUST(functional::FusedLstmCell(igates, hgates, cx, params.b_ih(), params.b_hh()));

      auto outputs = std::make_shared<TensorTuple>(2);
      (*outputs)[0] = JUST(params.matmul_hr((*result)[0]));
      (*outputs)[1] = (*result)[1];
      return outputs;
    }

    std::shared_ptr<one::Tensor> gates = JUST(params.linear_hh(hx));
    if (pre_compute_input) {
      gates = JUST(functional::Add(gates, input, 1.0, true));
    } else {
      std::shared_ptr<one::Tensor> gates_ih = JUST(params.linear_ih(input));
      gates = JUST(functional::Add(gates, gates_ih, 1.0, true));
    }
    std::shared_ptr<one::TensorTuple> chunked_gates = JUST(functional::Chunk(gates, 4, 1));
    std::shared_ptr<one::Tensor> ingate = JUST(functional::Sigmoid((*chunked_gates)[0]));
    std::shared_ptr<one::Tensor> forgetgate = JUST(functional::Sigmoid((*chunked_gates)[1]));
    std::shared_ptr<one::Tensor> cellgate = JUST(functional::Tanh((*chunked_gates)[2]));
    std::shared_ptr<one::Tensor> outgate = JUST(functional::Sigmoid((*chunked_gates)[3]));
    std::shared_ptr<one::Tensor> cy = JUST(functional::Mul(forgetgate, cx));
    cellgate = JUST(functional::Mul(ingate, cellgate));
    cy = JUST(functional::Add(cy, cellgate, 1.0, true));
    std::shared_ptr<one::Tensor> tanh_cy = JUST(functional::Tanh(cy));
    std::shared_ptr<one::Tensor> hy = JUST(functional::Mul(outgate, tanh_cy));
    auto outputs = std::make_shared<TensorTuple>(2);
    (*outputs)[0] = JUST(params.matmul_hr(hy));
    (*outputs)[1] = cy;
    return outputs;
  }
};

class RnnTanhCellFunctor {
 public:
  RnnTanhCellFunctor() {}

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& hx,
                           const std::shared_ptr<one::Tensor>& w_ih,
                           const std::shared_ptr<one::Tensor>& w_hh,
                           const Optional<one::Tensor>& b_ih,
                           const Optional<one::Tensor>& b_hh) const {
    JUST(check_rnn_cell_forward_input(input, w_ih->shape()->At(1)));
    JUST(check_rnn_cell_forward_hidden(input, hx, w_hh->shape()->At(1), 0));
    std::shared_ptr<one::Tensor> bias_ih = nullptr;
    std::shared_ptr<one::Tensor> bias_hh = nullptr;
    if (b_ih.has_value() && b_hh.has_value()) {
      bias_ih = JUST(b_ih);
      bias_hh = JUST(b_hh);
    }
    return SimpleCell<tanh_f, CellParams>{}(input, hx,
                                            CellParams{w_ih, w_hh, bias_ih, bias_hh, nullptr});
  }
};

class RnnReluCellFunctor {
 public:
  RnnReluCellFunctor() {}

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& hx,
                           const std::shared_ptr<one::Tensor>& w_ih,
                           const std::shared_ptr<one::Tensor>& w_hh,
                           const Optional<one::Tensor>& b_ih,
                           const Optional<one::Tensor>& b_hh) const {
    JUST(check_rnn_cell_forward_input(input, w_ih->shape()->At(1)));
    JUST(check_rnn_cell_forward_hidden(input, hx, w_hh->shape()->At(1), 0));
    std::shared_ptr<one::Tensor> bias_ih = nullptr;
    std::shared_ptr<one::Tensor> bias_hh = nullptr;
    if (b_ih.has_value() && b_hh.has_value()) {
      bias_ih = JUST(b_ih);
      bias_hh = JUST(b_hh);
    }
    return SimpleCell<relu_f, CellParams>{}(input, hx,
                                            CellParams{w_ih, w_hh, bias_ih, bias_hh, nullptr});
  }
};

class GruCellFunctor {
 public:
  GruCellFunctor() {}

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& hx,
                           const std::shared_ptr<one::Tensor>& w_ih,
                           const std::shared_ptr<one::Tensor>& w_hh,
                           const Optional<one::Tensor>& b_ih,
                           const Optional<one::Tensor>& b_hh) const {
    JUST(check_rnn_cell_forward_input(input, w_ih->shape()->At(1)));
    JUST(check_rnn_cell_forward_hidden(input, hx, w_hh->shape()->At(1), 0));
    std::shared_ptr<one::Tensor> bias_ih = nullptr;
    std::shared_ptr<one::Tensor> bias_hh = nullptr;
    if (b_ih.has_value() && b_hh.has_value()) {
      bias_ih = JUST(b_ih);
      bias_hh = JUST(b_hh);
    }
    return GRUCell<CellParams>{}(input, hx, CellParams{w_ih, w_hh, bias_ih, bias_hh, nullptr});
  }
};

class LstmCellFunctor {
 public:
  LstmCellFunctor() {}

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& input,
                                const one::TensorTuple& hx,
                                const std::shared_ptr<one::Tensor>& w_ih,
                                const std::shared_ptr<one::Tensor>& w_hh,
                                const Optional<one::Tensor>& b_ih,
                                const Optional<one::Tensor>& b_hh) const {
    CHECK_OR_RETURN(hx.size() == 2) << "lstm_cell expects two hidden states";
    JUST(check_rnn_cell_forward_input(input, w_ih->shape()->At(1)));
    auto hidden_size = w_hh->shape()->At(1);
    JUST(check_rnn_cell_forward_hidden(input, hx[0], hidden_size, 0));
    JUST(check_rnn_cell_forward_hidden(input, hx[1], hidden_size, 0));
    std::shared_ptr<one::Tensor> bias_ih = nullptr;
    std::shared_ptr<one::Tensor> bias_hh = nullptr;
    if (b_ih.has_value() && b_hh.has_value()) {
      bias_ih = JUST(b_ih);
      bias_hh = JUST(b_hh);
    }
    return LSTMCell<CellParams>{}(input, hx, CellParams{w_ih, w_hh, bias_ih, bias_hh, nullptr});
  }
};

class FusedGruCellFunctor {
 public:
  FusedGruCellFunctor() {
    op_with_bias_ = CHECK_JUST(one::OpBuilder("fused_gru_cell")
                                   .Input("input_gates")
                                   .Input("hidden_gates")
                                   .Input("hx")
                                   .Input("input_bias")
                                   .Input("hidden_bias")
                                   .Output("hy")
                                   .Output("workspace")
                                   .Build());
    op_without_bias_ = CHECK_JUST(one::OpBuilder("fused_gru_cell")
                                      .Input("input_gates")
                                      .Input("hidden_gates")
                                      .Input("hx")
                                      .Output("hy")
                                      .Output("workspace")
                                      .Build());
  }

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& igates,
                                const std::shared_ptr<one::Tensor>& hgates,
                                const std::shared_ptr<one::Tensor>& hx,
                                const Optional<one::Tensor>& b_ih,
                                const Optional<one::Tensor>& b_hh) const {
    std::shared_ptr<TensorTuple> kernel_result;
    if (b_ih.has_value() && b_hh.has_value()) {
      kernel_result = JUST(OpInterpUtil::Dispatch<TensorTuple>(
          *op_with_bias_, {igates, hgates, hx, JUST(b_ih), JUST(b_hh)}));
    } else {
      kernel_result =
          JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_without_bias_, {igates, hgates, hx}));
    }
    return kernel_result;
  }

 private:
  std::shared_ptr<OpExpr> op_with_bias_;
  std::shared_ptr<OpExpr> op_without_bias_;
};

class FusedGruCellGradFunctor {
 public:
  FusedGruCellGradFunctor() {
    op_with_bias_ = CHECK_JUST(one::OpBuilder("fused_gru_cell_grad")
                                   .Input("grad_hy")
                                   .Input("workspace")
                                   .Output("grad_input_gates")
                                   .Output("grad_hidden_gates")
                                   .Output("grad_hx")
                                   .Output("grad_input_bias")
                                   .Output("grad_hidden_bias")
                                   .Build());
    op_with_bias_without_hx_ = CHECK_JUST(one::OpBuilder("fused_gru_cell_grad")
                                              .Input("grad_hy")
                                              .Input("workspace")
                                              .Output("grad_input_gates")
                                              .Output("grad_hidden_gates")
                                              .Output("grad_input_bias")
                                              .Output("grad_hidden_bias")
                                              .Build());
    op_without_bias_ = CHECK_JUST(one::OpBuilder("fused_gru_cell_grad")
                                      .Input("grad_hy")
                                      .Input("workspace")
                                      .Output("grad_input_gates")
                                      .Output("grad_hidden_gates")
                                      .Output("grad_hx")
                                      .Build());
    op_without_bias_without_hx_ = CHECK_JUST(one::OpBuilder("fused_gru_cell_grad")
                                                 .Input("grad_hy")
                                                 .Input("workspace")
                                                 .Output("grad_input_gates")
                                                 .Output("grad_hidden_gates")
                                                 .Build());
  }

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& grad_hy,
                                const std::shared_ptr<one::Tensor>& workspace, bool has_bias,
                                bool hx_needs_grad) const {
    std::shared_ptr<TensorTuple> kernel_result;
    if (has_bias) {
      if (hx_needs_grad) {
        kernel_result =
            JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_with_bias_, {grad_hy, workspace}));
      } else {
        kernel_result = JUST(
            OpInterpUtil::Dispatch<TensorTuple>(*op_with_bias_without_hx_, {grad_hy, workspace}));
      }
    } else {
      if (hx_needs_grad) {
        kernel_result =
            JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_without_bias_, {grad_hy, workspace}));
      } else {
        kernel_result = JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_without_bias_without_hx_,
                                                                 {grad_hy, workspace}));
      }
    }
    return kernel_result;
  }

 private:
  std::shared_ptr<OpExpr> op_with_bias_;
  std::shared_ptr<OpExpr> op_with_bias_without_hx_;
  std::shared_ptr<OpExpr> op_without_bias_;
  std::shared_ptr<OpExpr> op_without_bias_without_hx_;
};

class FusedLstmCellFunctor {
 public:
  FusedLstmCellFunctor() {
    op_with_bias_ = CHECK_JUST(one::OpBuilder("fused_lstm_cell")
                                   .Input("input_gates")
                                   .Input("hidden_gates")
                                   .Input("cx")
                                   .Input("input_bias")
                                   .Input("hidden_bias")
                                   .Output("hy")
                                   .Output("cy")
                                   .Output("workspace")
                                   .Build());
    op_without_bias_ = CHECK_JUST(one::OpBuilder("fused_lstm_cell")
                                      .Input("input_gates")
                                      .Input("hidden_gates")
                                      .Input("cx")
                                      .Output("hy")
                                      .Output("cy")
                                      .Output("workspace")
                                      .Build());
  }

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& igates,
                                const std::shared_ptr<one::Tensor>& hgates,
                                const std::shared_ptr<one::Tensor>& cx,
                                const Optional<one::Tensor>& b_ih,
                                const Optional<one::Tensor>& b_hh) const {
    std::shared_ptr<TensorTuple> kernel_result;
    if (b_ih.has_value() && b_hh.has_value()) {
      kernel_result = JUST(OpInterpUtil::Dispatch<TensorTuple>(
          *op_with_bias_, {igates, hgates, cx, JUST(b_ih), JUST(b_hh)}));
    } else {
      kernel_result =
          JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_without_bias_, {igates, hgates, cx}));
    }
    return kernel_result;
  }

 private:
  std::shared_ptr<OpExpr> op_with_bias_;
  std::shared_ptr<OpExpr> op_without_bias_;
};

class FusedLstmCellGradFunctor {
 public:
  FusedLstmCellGradFunctor() {
    op_with_bias_ = CHECK_JUST(one::OpBuilder("fused_lstm_cell_grad")
                                   .Input("grad_hy")
                                   .Input("grad_cy")
                                   .Input("cx")
                                   .Input("cy")
                                   .Input("workspace")
                                   .Output("grad_gates")
                                   .Output("grad_cx")
                                   .Output("grad_bias")
                                   .Build());
    op_without_bias_ = CHECK_JUST(one::OpBuilder("fused_lstm_cell_grad")
                                      .Input("grad_hy")
                                      .Input("grad_cy")
                                      .Input("cx")
                                      .Input("cy")
                                      .Input("workspace")
                                      .Output("grad_gates")
                                      .Output("grad_cx")
                                      .Build());
    op_with_bias_no_grad_cx_ = CHECK_JUST(one::OpBuilder("fused_lstm_cell_grad")
                                              .Input("grad_hy")
                                              .Input("grad_cy")
                                              .Input("cx")
                                              .Input("cy")
                                              .Input("workspace")
                                              .Output("grad_gates")
                                              .Output("grad_bias")
                                              .Build());
    op_without_bias_no_grad_cx_ = CHECK_JUST(one::OpBuilder("fused_lstm_cell_grad")
                                                 .Input("grad_hy")
                                                 .Input("grad_cy")
                                                 .Input("cx")
                                                 .Input("cy")
                                                 .Input("workspace")
                                                 .Output("grad_gates")
                                                 .Build());
  }

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& grad_hy,
                                const std::shared_ptr<one::Tensor>& grad_cy,
                                const std::shared_ptr<one::Tensor>& cx,
                                const std::shared_ptr<one::Tensor>& cy,
                                const std::shared_ptr<one::Tensor>& workspace, bool need_cx_grad,
                                bool has_bias) const {
    std::shared_ptr<TensorTuple> kernel_result;
    if (has_bias) {
      if (need_cx_grad) {
        kernel_result = JUST(OpInterpUtil::Dispatch<TensorTuple>(
            *op_with_bias_, {grad_hy, grad_cy, cx, cy, workspace}));
      } else {
        kernel_result = JUST(OpInterpUtil::Dispatch<TensorTuple>(
            *op_with_bias_no_grad_cx_, {grad_hy, grad_cy, cx, cy, workspace}));
      }
    } else {
      if (need_cx_grad) {
        kernel_result = JUST(OpInterpUtil::Dispatch<TensorTuple>(
            *op_without_bias_, {grad_hy, grad_cy, cx, cy, workspace}));
      } else {
        kernel_result = JUST(OpInterpUtil::Dispatch<TensorTuple>(
            *op_without_bias_no_grad_cx_, {grad_hy, grad_cy, cx, cy, workspace}));
      }
    }
    return kernel_result;
  }

 private:
  std::shared_ptr<OpExpr> op_with_bias_;
  std::shared_ptr<OpExpr> op_with_bias_no_grad_cx_;
  std::shared_ptr<OpExpr> op_without_bias_;
  std::shared_ptr<OpExpr> op_without_bias_no_grad_cx_;
};

template<typename cell_type>
Maybe<TensorTuple> _rnn_impl(const std::shared_ptr<one::Tensor>& input,
                             const std::shared_ptr<one::Tensor>& hx, const one::TensorTuple& params,
                             const bool& has_biases, const int32_t& num_layers,
                             const float& dropout, const bool& train, const bool& bidirectional,
                             const bool& batch_first) {
  TensorTuple hiddens;
  hiddens.emplace_back(hx);
  JUST(check_attributes(input, params, hiddens));

  std::shared_ptr<one::Tensor> rnn_input = input;
  if (batch_first) {
    std::vector<int32_t> dims = {1, 0, 2};
    rnn_input = JUST(functional::Permute(input, dims));
  }
  auto rnn_params = JUST(gather_params(params, has_biases));
  std::shared_ptr<TensorTuple> rnn_hiddens = JUST(functional::Unbind(hx, 0));
  std::shared_ptr<TensorTuple> rnn_inputs = JUST(functional::Unbind(rnn_input, 0));

  auto generator = JUST(one::DefaultAutoGenerator());

  TensorTuple final_hiddens;
  if (bidirectional) {
    std::shared_ptr<TensorTuple> fw_outputs = std::make_shared<TensorTuple>(rnn_inputs->size());
    std::shared_ptr<TensorTuple> bw_outputs = std::make_shared<TensorTuple>(rnn_inputs->size());
    for (int32_t l = 0; l < num_layers; ++l) {
      // forward direction
      std::shared_ptr<one::Tensor> fw_hidden = (*rnn_hiddens)[l * 2];
      auto& fw_cell_param = (*rnn_params)[l * 2];
      for (int32_t i = 0; i < rnn_inputs->size(); ++i) {
        fw_hidden = JUST(cell_type{}((*rnn_inputs)[i], fw_hidden, fw_cell_param));
        (*fw_outputs)[i] = fw_hidden;
      }
      final_hiddens.emplace_back(fw_hidden);

      // reverse direction
      std::shared_ptr<one::Tensor> bw_hidden = (*rnn_hiddens)[l * 2 + 1];
      auto& bw_cell_param = (*rnn_params)[l * 2 + 1];
      for (int32_t i = rnn_inputs->size() - 1; i >= 0; i--) {
        bw_hidden = JUST(cell_type{}((*rnn_inputs)[i], bw_hidden, bw_cell_param));
        (*bw_outputs)[i] = bw_hidden;
      }
      final_hiddens.emplace_back(bw_hidden);

      // concat fw_outputs and bw_outputs
      for (int32_t i = 0; i < rnn_inputs->size(); ++i) {
        (*rnn_inputs)[i] = JUST(functional::Concat({(*fw_outputs)[i], (*bw_outputs)[i]},
                                                   bw_hidden->shape()->NumAxes() - 1));
      }

      if (dropout != 0 && train && l < num_layers - 1) {
        std::shared_ptr<one::Tensor> stack_res = JUST(functional::Stack(*rnn_inputs, 0));
        std::shared_ptr<one::Tensor> dropout_res =
            JUST(functional::Dropout(stack_res, dropout, train, false, generator, nullptr));
        rnn_inputs = JUST(functional::Unbind(dropout_res, 0));
      }
    }
  } else {
    for (int32_t l = 0; l < num_layers; ++l) {
      std::shared_ptr<one::Tensor> hidden = (*rnn_hiddens)[l];
      auto& cell_param = (*rnn_params)[l];
      for (int32_t i = 0; i < rnn_inputs->size(); ++i) {
        hidden = JUST(cell_type{}((*rnn_inputs)[i], hidden, cell_param));
        (*rnn_inputs)[i] = hidden;
      }
      final_hiddens.emplace_back(hidden);
      if (dropout != 0 && train && l < num_layers - 1) {
        std::shared_ptr<one::Tensor> stack_res = JUST(functional::Stack(*rnn_inputs, 0));
        std::shared_ptr<one::Tensor> dropout_res =
            JUST(functional::Dropout(stack_res, dropout, train, false, generator, nullptr));
        rnn_inputs = JUST(functional::Unbind(dropout_res, 0));
      }
    }
  }

  TensorTuple output;
  std::shared_ptr<one::Tensor> output_0 = JUST(functional::Stack(*rnn_inputs, 0));
  if (batch_first) {
    std::vector<int32_t> dims = {1, 0, 2};
    output.emplace_back(JUST(functional::Permute(output_0, dims)));
  } else {
    output.emplace_back(output_0);
  }
  output.emplace_back(JUST(functional::Stack(final_hiddens, 0)));
  return output;
}

class RnnTanhInputFunctor {
 public:
  RnnTanhInputFunctor() {}

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& input,
                                const std::shared_ptr<one::Tensor>& hx,
                                const one::TensorTuple& params, const bool& has_biases,
                                const int32_t& num_layers, const float& dropout, const bool& train,
                                const bool& bidirectional, const bool& batch_first) const {
    return _rnn_impl<SimpleCell<tanh_f, CellParams>>(input, hx, params, has_biases, num_layers,
                                                     dropout, train, bidirectional, batch_first);
  }
};

class RnnReluInputFunctor {
 public:
  RnnReluInputFunctor() {}

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& input,
                                const std::shared_ptr<one::Tensor>& hx,
                                const one::TensorTuple& params, const bool& has_biases,
                                const int32_t& num_layers, const float& dropout, const bool& train,
                                const bool& bidirectional, const bool& batch_first) const {
    return _rnn_impl<SimpleCell<relu_f, CellParams>>(input, hx, params, has_biases, num_layers,
                                                     dropout, train, bidirectional, batch_first);
  }
};

template<typename cell_type>
Maybe<TensorTuple> _rnn_pack_sequence_impl(const std::shared_ptr<one::Tensor>& input,
                                           const std::shared_ptr<one::Tensor>& batch_sizes,
                                           const std::shared_ptr<one::Tensor>& hx,
                                           const one::TensorTuple& params, const bool& has_biases,
                                           const int32_t& num_layers, const float& dropout,
                                           const bool& train, const bool& bidirectional) {
  auto rnn_params = JUST(gather_params(params, has_biases));
  std::shared_ptr<TensorTuple> rnn_hiddens = JUST(functional::Unbind(hx, 0));
  auto generator = JUST(one::DefaultAutoGenerator());

  TensorTuple final_hiddens;

  std::vector<int64_t> batch_sizes_vec;
  batch_sizes_vec.resize(batch_sizes->nelement());
  const auto& callback = [&](ep::Stream* stream,
                             const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
    SyncAutoMemcpy(stream, batch_sizes_vec.data(), eager_blob_object->dptr(),
                   batch_sizes_vec.size() * sizeof(int64_t), memory::MakeHostMemCase(),
                   eager_blob_object->mem_case());
  };
  JUST(SyncAccessTensorWithTimeOut(batch_sizes, callback, "const"));
  int64_t num_steps = batch_sizes->shape()->At(0);
  std::shared_ptr<TensorTuple> rnn_inputs = std::make_shared<TensorTuple>(num_steps);
  int64_t input_offset = 0;
  for (int32_t i = 0; i < num_steps; ++i) {
    const int64_t batch_size = batch_sizes_vec[i];
    (*rnn_inputs)[i] = JUST(functional::Narrow(input, 0, input_offset, batch_size));
    input_offset += batch_size;
  }

  if (bidirectional) {
    std::shared_ptr<TensorTuple> fw_outputs = std::make_shared<TensorTuple>(rnn_inputs->size());
    std::shared_ptr<TensorTuple> bw_outputs = std::make_shared<TensorTuple>(rnn_inputs->size());
    for (int32_t l = 0; l < num_layers; ++l) {
      // forward direction
      int64_t last_batch_size = batch_sizes_vec[0];
      std::shared_ptr<one::Tensor> fw_hidden = (*rnn_hiddens)[l * 2];
      auto& fw_cell_param = (*rnn_params)[l * 2];

      TensorTuple fw_final_hiddens_for_single_layer;
      for (int32_t i = 0; i < num_steps; ++i) {
        const int64_t batch_size = batch_sizes_vec[i];
        const int64_t dec = last_batch_size - batch_size;
        if (dec > 0) {
          fw_final_hiddens_for_single_layer.emplace_back(
              JUST(functional::Narrow(fw_hidden, 0, last_batch_size - dec, dec)));
          fw_hidden = JUST(functional::Narrow(fw_hidden, 0, 0, last_batch_size - dec));
        }
        last_batch_size = batch_size;
        fw_hidden = JUST(cell_type{}((*rnn_inputs)[i], fw_hidden, fw_cell_param));
        (*fw_outputs)[i] = fw_hidden;
      }
      fw_final_hiddens_for_single_layer.emplace_back(fw_hidden);
      std::reverse(fw_final_hiddens_for_single_layer.begin(),
                   fw_final_hiddens_for_single_layer.end());
      final_hiddens.emplace_back(JUST(functional::Concat(fw_final_hiddens_for_single_layer, 0)));

      // reverse direction
      last_batch_size = batch_sizes_vec[num_steps - 1];
      std::shared_ptr<one::Tensor> bw_hidden =
          JUST(functional::Narrow((*rnn_hiddens)[l * 2 + 1], 0, 0, last_batch_size));
      auto& bw_cell_param = (*rnn_params)[l * 2 + 1];
      // Here the situation is similar to that above, except we start out with
      // the smallest batch size (and a small set of hidden states we actually use),
      // and progressively expand the hidden states, as we move backwards over the
      // 1D list of inputs.
      for (int64_t i = num_steps - 1; i >= 0; --i) {
        const int64_t batch_size = batch_sizes_vec[i];
        const int64_t inc = batch_size - last_batch_size;
        if (inc > 0) {
          std::shared_ptr<one::Tensor> hidden_slice = JUST(functional::Narrow(
              (*rnn_hiddens)[l * 2 + 1], 0, last_batch_size, batch_size - last_batch_size));
          std::shared_ptr<TensorTuple> tmp = std::make_shared<TensorTuple>(2);
          (*tmp)[0] = bw_hidden;
          (*tmp)[1] = hidden_slice;
          bw_hidden = JUST(functional::Concat(*tmp, 0));
        }
        last_batch_size = batch_size;
        bw_hidden = JUST(cell_type{}((*rnn_inputs)[i], bw_hidden, bw_cell_param));
        (*bw_outputs)[i] = bw_hidden;
      }

      final_hiddens.emplace_back(bw_hidden);

      // concat fw_outputs and bw_outputs
      for (int32_t i = 0; i < num_steps; ++i) {
        (*rnn_inputs)[i] = JUST(functional::Concat({(*fw_outputs)[i], (*bw_outputs)[i]},
                                                   bw_hidden->shape()->NumAxes() - 1));
      }

      if (dropout != 0 && train && l < num_layers - 1) {
        std::shared_ptr<one::Tensor> stack_res = JUST(functional::Concat(*rnn_inputs, 0));
        std::shared_ptr<one::Tensor> dropout_res =
            JUST(functional::Dropout(stack_res, dropout, train, false, generator, nullptr));
        int64_t input_offset = 0;
        for (int32_t i = 0; i < num_steps; ++i) {
          const int64_t batch_size = batch_sizes_vec[i];
          (*rnn_inputs)[i] = JUST(functional::Narrow(dropout_res, 0, input_offset, batch_size));
          input_offset += batch_size;
        }
      }
    }
  } else {
    // Batch sizes is a sequence of decreasing lengths, which are offsets
    // into a 1D list of inputs. At every step we slice out batch_size elements,
    // and possibly account for the decrease in the batch size since the last step,
    // which requires us to slice the hidden state (since some sequences
    // are completed now). The sliced parts are also saved, because we will need
    // to return a tensor of final hidden state.
    for (int32_t l = 0; l < num_layers; ++l) {
      int64_t last_batch_size = batch_sizes_vec[0];
      std::shared_ptr<one::Tensor> hidden = (*rnn_hiddens)[l];
      auto& cell_param = (*rnn_params)[l];
      TensorTuple final_hiddens_for_single_layer;
      for (int32_t i = 0; i < num_steps; ++i) {
        const int64_t batch_size = batch_sizes_vec[i];
        const int64_t dec = last_batch_size - batch_size;
        if (dec > 0) {
          final_hiddens_for_single_layer.emplace_back(
              JUST(functional::Narrow(hidden, 0, last_batch_size - dec, dec)));
          hidden = JUST(functional::Narrow(hidden, 0, 0, last_batch_size - dec));
        }
        last_batch_size = batch_size;
        hidden = JUST(cell_type{}((*rnn_inputs)[i], hidden, cell_param));
        (*rnn_inputs)[i] = hidden;
      }
      final_hiddens_for_single_layer.emplace_back(hidden);
      std::reverse(final_hiddens_for_single_layer.begin(), final_hiddens_for_single_layer.end());
      final_hiddens.emplace_back(JUST(functional::Concat(final_hiddens_for_single_layer, 0)));

      if (dropout != 0 && train && l < num_layers - 1) {
        std::shared_ptr<one::Tensor> stack_res = JUST(functional::Concat(*rnn_inputs, 0));
        std::shared_ptr<one::Tensor> dropout_res =
            JUST(functional::Dropout(stack_res, dropout, train, false, generator, nullptr));
        int64_t input_offset = 0;
        for (int32_t i = 0; i < num_steps; ++i) {
          const int64_t batch_size = batch_sizes_vec[i];
          (*rnn_inputs)[i] = JUST(functional::Narrow(dropout_res, 0, input_offset, batch_size));
          input_offset += batch_size;
        }
      }
    }
  }

  TensorTuple output;
  output.emplace_back(JUST(functional::Concat(*rnn_inputs, 0)));
  output.emplace_back(JUST(functional::Stack(final_hiddens, 0)));
  return output;
}

class RnnTanhDataFunctor {
 public:
  RnnTanhDataFunctor() {}

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& data,
                                const std::shared_ptr<one::Tensor>& batch_sizes,
                                const std::shared_ptr<one::Tensor>& hx,
                                const one::TensorTuple& params, const bool& has_biases,
                                const int32_t& num_layers, const float& dropout, const bool& train,
                                const bool& bidirectional) const {
    return _rnn_pack_sequence_impl<SimpleCell<tanh_f, CellParams>>(
        data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
  }
};

class RnnReluDataFunctor {
 public:
  RnnReluDataFunctor() {}

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& data,
                                const std::shared_ptr<one::Tensor>& batch_sizes,
                                const std::shared_ptr<one::Tensor>& hx,
                                const one::TensorTuple& params, const bool& has_biases,
                                const int32_t& num_layers, const float& dropout, const bool& train,
                                const bool& bidirectional) const {
    return _rnn_pack_sequence_impl<SimpleCell<relu_f, CellParams>>(
        data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
  }
};

Maybe<TensorTuple> _lstm_impl(const std::shared_ptr<one::Tensor>& input, const one::TensorTuple& hx,
                              const one::TensorTuple& params, const bool& has_biases,
                              const int32_t& num_layers, const float& dropout, const bool& train,
                              const bool& bidirectional, const bool& batch_first) {
  CHECK_OR_RETURN(hx.size() == 2) << "lstm expects two hidden states";
  // if cells are of different size, that means projections are used
  bool has_projections = (hx[0]->shape()->At(2) != hx[1]->shape()->At(2));
  JUST(check_attributes(input, params, hx));
  std::shared_ptr<one::Tensor> rnn_input = input;
  if (batch_first) {
    std::vector<int32_t> dims = {1, 0, 2};
    rnn_input = JUST(functional::Permute(input, dims));
  }
  auto rnn_params = JUST(gather_params(params, has_biases, has_projections));

  std::shared_ptr<TensorTuple> layer_hxs = JUST(functional::Unbind(hx[0], 0));
  std::shared_ptr<TensorTuple> layer_cxs = JUST(functional::Unbind(hx[1], 0));
  std::shared_ptr<TensorTuple> rnn_inputs = JUST(functional::Unbind(rnn_input, 0));

  auto generator = JUST(one::DefaultAutoGenerator());

  TensorTuple final_hy;
  TensorTuple final_cy;

  if (bidirectional) {
    std::shared_ptr<TensorTuple> fw_outputs = std::make_shared<TensorTuple>(rnn_inputs->size());
    std::shared_ptr<TensorTuple> lstm_cell_out = std::make_shared<TensorTuple>(2);
    std::shared_ptr<TensorTuple> bw_outputs = std::make_shared<TensorTuple>(rnn_inputs->size());

    for (int32_t l = 0; l < num_layers; ++l) {
      // forward direction
      (*lstm_cell_out)[0] = (*layer_hxs)[l * 2];
      (*lstm_cell_out)[1] = (*layer_cxs)[l * 2];
      auto& fw_cell_param = (*rnn_params)[l * 2];
      for (int32_t i = 0; i < rnn_inputs->size(); ++i) {
        lstm_cell_out =
            JUST(LSTMCell<CellParams>{}((*rnn_inputs)[i], *lstm_cell_out, fw_cell_param));
        (*fw_outputs)[i] = (*lstm_cell_out)[0];
      }
      final_hy.emplace_back((*lstm_cell_out)[0]);
      final_cy.emplace_back((*lstm_cell_out)[1]);

      // reverse direction
      (*lstm_cell_out)[0] = (*layer_hxs)[l * 2 + 1];
      (*lstm_cell_out)[1] = (*layer_cxs)[l * 2 + 1];
      auto& bw_cell_param = (*rnn_params)[l * 2 + 1];
      for (int32_t i = rnn_inputs->size() - 1; i >= 0; i--) {
        lstm_cell_out =
            JUST(LSTMCell<CellParams>{}((*rnn_inputs)[i], *lstm_cell_out, bw_cell_param));
        (*bw_outputs)[i] = (*lstm_cell_out)[0];
      }
      final_hy.emplace_back((*lstm_cell_out)[0]);
      final_cy.emplace_back((*lstm_cell_out)[1]);

      // concat fw_outputs and bw_outputs
      for (int32_t i = 0; i < rnn_inputs->size(); ++i) {
        (*rnn_inputs)[i] = JUST(functional::Concat({(*fw_outputs)[i], (*bw_outputs)[i]},
                                                   (*bw_outputs)[0]->shape()->NumAxes() - 1));
      }

      if (dropout != 0 && train && l < num_layers - 1) {
        std::shared_ptr<one::Tensor> stack_res = JUST(functional::Stack(*rnn_inputs, 0));
        std::shared_ptr<one::Tensor> dropout_res =
            JUST(functional::Dropout(stack_res, dropout, train, false, generator, nullptr));
        rnn_inputs = JUST(functional::Unbind(dropout_res, 0));
      }
    }

  } else {
    std::shared_ptr<TensorTuple> lstm_cell_out = std::make_shared<TensorTuple>(2);

    for (int32_t l = 0; l < num_layers; ++l) {
      auto& cell_param = (*rnn_params)[l];
      (*lstm_cell_out)[0] = (*layer_hxs)[l];
      (*lstm_cell_out)[1] = (*layer_cxs)[l];
      for (int32_t i = 0; i < rnn_inputs->size(); ++i) {
        lstm_cell_out = JUST(LSTMCell<CellParams>{}((*rnn_inputs)[i], *lstm_cell_out, cell_param));
        (*rnn_inputs)[i] = (*lstm_cell_out)[0];
      }
      final_hy.emplace_back((*lstm_cell_out)[0]);
      final_cy.emplace_back((*lstm_cell_out)[1]);

      if (dropout != 0 && train && l < num_layers - 1) {
        std::shared_ptr<one::Tensor> stack_res = JUST(functional::Stack(*rnn_inputs, 0));
        std::shared_ptr<one::Tensor> dropout_res =
            JUST(functional::Dropout(stack_res, dropout, train, false, generator, nullptr));
        rnn_inputs = JUST(functional::Unbind(dropout_res, 0));
      }
    }
  }

  TensorTuple output;
  std::shared_ptr<one::Tensor> output_0 = JUST(functional::Stack(*rnn_inputs, 0));
  if (batch_first) {
    std::vector<int32_t> dims = {1, 0, 2};
    output.emplace_back(JUST(functional::Permute(output_0, dims)));
  } else {
    output.emplace_back(output_0);
  }
  output.emplace_back(JUST(functional::Stack(final_hy, 0)));
  output.emplace_back(JUST(functional::Stack(final_cy, 0)));
  return output;
}

class LstmInputFunctor {
 public:
  LstmInputFunctor() {}

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& input,
                                const one::TensorTuple& hx, const one::TensorTuple& params,
                                const bool& has_biases, const int32_t& num_layers,
                                const float& dropout, const bool& train, const bool& bidirectional,
                                const bool& batch_first) const {
    return _lstm_impl(input, hx, params, has_biases, num_layers, dropout, train, bidirectional,
                      batch_first);
  }
};

Maybe<TensorTuple> _lstm_pack_sequence_impl(const std::shared_ptr<one::Tensor>& input,
                                            const std::shared_ptr<one::Tensor>& batch_sizes,
                                            const one::TensorTuple& hx,
                                            const one::TensorTuple& params, const bool& has_biases,
                                            const int32_t& num_layers, const float& dropout,
                                            const bool& train, const bool& bidirectional) {
  CHECK_OR_RETURN(hx.size() == 2) << "lstm expects two hidden states";
  // if cells are of different size, that means projections are used
  bool has_projections = (hx[0]->shape()->At(2) != hx[1]->shape()->At(2));
  auto rnn_params = JUST(gather_params(params, has_biases, has_projections));

  std::shared_ptr<TensorTuple> layer_hxs = JUST(functional::Unbind(hx[0], 0));
  std::shared_ptr<TensorTuple> layer_cxs = JUST(functional::Unbind(hx[1], 0));

  std::vector<int64_t> batch_sizes_vec;
  batch_sizes_vec.resize(batch_sizes->nelement());
  const auto& callback = [&](ep::Stream* stream,
                             const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
    SyncAutoMemcpy(stream, batch_sizes_vec.data(), eager_blob_object->dptr(),
                   batch_sizes_vec.size() * sizeof(int64_t), memory::MakeHostMemCase(),
                   eager_blob_object->mem_case());
  };
  JUST(SyncAccessTensorWithTimeOut(batch_sizes, callback, "const"));
  int64_t num_steps = batch_sizes->shape()->At(0);
  std::shared_ptr<TensorTuple> rnn_inputs = std::make_shared<TensorTuple>(num_steps);
  int64_t input_offset = 0;
  for (int32_t i = 0; i < num_steps; ++i) {
    const int64_t batch_size = batch_sizes_vec[i];
    (*rnn_inputs)[i] = JUST(functional::Narrow(input, 0, input_offset, batch_size));
    input_offset += batch_size;
  }

  auto generator = JUST(one::DefaultAutoGenerator());

  TensorTuple final_hy;
  TensorTuple final_cy;

  if (bidirectional) {
    std::shared_ptr<TensorTuple> fw_outputs = std::make_shared<TensorTuple>(rnn_inputs->size());
    std::shared_ptr<TensorTuple> lstm_cell_out = std::make_shared<TensorTuple>(2);
    std::shared_ptr<TensorTuple> bw_outputs = std::make_shared<TensorTuple>(rnn_inputs->size());

    for (int32_t l = 0; l < num_layers; ++l) {
      int64_t last_batch_size = batch_sizes_vec[0];
      // forward direction
      (*lstm_cell_out)[0] = (*layer_hxs)[l * 2];
      (*lstm_cell_out)[1] = (*layer_cxs)[l * 2];
      auto& fw_cell_param = (*rnn_params)[l * 2];

      TensorTuple final_hy_for_single_layer;
      TensorTuple final_cy_for_single_layer;
      for (int32_t i = 0; i < num_steps; ++i) {
        const int64_t batch_size = batch_sizes_vec[i];
        const int64_t dec = last_batch_size - batch_size;
        if (dec > 0) {
          final_hy_for_single_layer.emplace_back(
              JUST(functional::Narrow((*lstm_cell_out)[0], 0, last_batch_size - dec, dec)));
          (*lstm_cell_out)[0] =
              JUST(functional::Narrow((*lstm_cell_out)[0], 0, 0, last_batch_size - dec));

          final_cy_for_single_layer.emplace_back(
              JUST(functional::Narrow((*lstm_cell_out)[1], 0, last_batch_size - dec, dec)));
          (*lstm_cell_out)[1] =
              JUST(functional::Narrow((*lstm_cell_out)[1], 0, 0, last_batch_size - dec));
        }
        last_batch_size = batch_size;
        lstm_cell_out =
            JUST(LSTMCell<CellParams>{}((*rnn_inputs)[i], *lstm_cell_out, fw_cell_param));
        (*fw_outputs)[i] = (*lstm_cell_out)[0];
      }
      final_hy_for_single_layer.emplace_back((*lstm_cell_out)[0]);
      final_cy_for_single_layer.emplace_back((*lstm_cell_out)[1]);
      std::reverse(final_hy_for_single_layer.begin(), final_hy_for_single_layer.end());
      std::reverse(final_cy_for_single_layer.begin(), final_cy_for_single_layer.end());
      final_hy.emplace_back(JUST(functional::Concat(final_hy_for_single_layer, 0)));
      final_cy.emplace_back(JUST(functional::Concat(final_cy_for_single_layer, 0)));

      // reverse direction
      last_batch_size = batch_sizes_vec[num_steps - 1];
      (*lstm_cell_out)[0] =
          JUST(functional::Narrow((*layer_hxs)[l * 2 + 1], 0, 0, last_batch_size));
      (*lstm_cell_out)[1] =
          JUST(functional::Narrow((*layer_cxs)[l * 2 + 1], 0, 0, last_batch_size));

      auto& bw_cell_param = (*rnn_params)[l * 2 + 1];

      for (int64_t i = num_steps - 1; i >= 0; --i) {
        const int64_t batch_size = batch_sizes_vec[i];
        const int64_t inc = batch_size - last_batch_size;
        if (inc > 0) {
          std::shared_ptr<one::Tensor> hxs_slice = JUST(functional::Narrow(
              (*layer_hxs)[l * 2 + 1], 0, last_batch_size, batch_size - last_batch_size));
          std::shared_ptr<TensorTuple> tmp = std::make_shared<TensorTuple>(2);
          (*tmp)[0] = (*lstm_cell_out)[0];
          (*tmp)[1] = hxs_slice;
          (*lstm_cell_out)[0] = JUST(functional::Concat(*tmp, 0));

          std::shared_ptr<one::Tensor> cxs_slice = JUST(functional::Narrow(
              (*layer_cxs)[l * 2 + 1], 0, last_batch_size, batch_size - last_batch_size));
          (*tmp)[0] = (*lstm_cell_out)[1];
          (*tmp)[1] = cxs_slice;
          (*lstm_cell_out)[1] = JUST(functional::Concat(*tmp, 0));
        }
        last_batch_size = batch_size;
        lstm_cell_out =
            JUST(LSTMCell<CellParams>{}((*rnn_inputs)[i], *lstm_cell_out, bw_cell_param));
        (*bw_outputs)[i] = (*lstm_cell_out)[0];
      }
      final_hy.emplace_back((*lstm_cell_out)[0]);
      final_cy.emplace_back((*lstm_cell_out)[1]);

      // concat fw_outputs and bw_outputs
      for (int32_t i = 0; i < rnn_inputs->size(); ++i) {
        (*rnn_inputs)[i] = JUST(functional::Concat({(*fw_outputs)[i], (*bw_outputs)[i]},
                                                   (*bw_outputs)[0]->shape()->NumAxes() - 1));
      }

      if (dropout != 0 && train && l < num_layers - 1) {
        std::shared_ptr<one::Tensor> stack_res = JUST(functional::Concat(*rnn_inputs, 0));
        std::shared_ptr<one::Tensor> dropout_res =
            JUST(functional::Dropout(stack_res, dropout, train, false, generator, nullptr));
        int64_t input_offset = 0;
        for (int32_t i = 0; i < num_steps; ++i) {
          const int64_t batch_size = batch_sizes_vec[i];
          (*rnn_inputs)[i] = JUST(functional::Narrow(dropout_res, 0, input_offset, batch_size));
          input_offset += batch_size;
        }
      }
    }
  } else {
    std::shared_ptr<TensorTuple> lstm_cell_out = std::make_shared<TensorTuple>(2);
    for (int32_t l = 0; l < num_layers; ++l) {
      int64_t last_batch_size = batch_sizes_vec[0];
      (*lstm_cell_out)[0] = (*layer_hxs)[l];
      (*lstm_cell_out)[1] = (*layer_cxs)[l];
      auto& cell_param = (*rnn_params)[l];
      TensorTuple final_hy_for_single_layer;
      TensorTuple final_cy_for_single_layer;
      for (int32_t i = 0; i < num_steps; ++i) {
        const int64_t batch_size = batch_sizes_vec[i];
        const int64_t dec = last_batch_size - batch_size;
        if (dec > 0) {
          final_hy_for_single_layer.emplace_back(
              JUST(functional::Narrow((*lstm_cell_out)[0], 0, last_batch_size - dec, dec)));
          (*lstm_cell_out)[0] =
              JUST(functional::Narrow((*lstm_cell_out)[0], 0, 0, last_batch_size - dec));

          final_cy_for_single_layer.emplace_back(
              JUST(functional::Narrow((*lstm_cell_out)[1], 0, last_batch_size - dec, dec)));
          (*lstm_cell_out)[1] =
              JUST(functional::Narrow((*lstm_cell_out)[1], 0, 0, last_batch_size - dec));
        }
        last_batch_size = batch_size;
        lstm_cell_out = JUST(LSTMCell<CellParams>{}((*rnn_inputs)[i], *lstm_cell_out, cell_param));
        (*rnn_inputs)[i] = (*lstm_cell_out)[0];
      }
      final_hy_for_single_layer.emplace_back((*lstm_cell_out)[0]);
      final_cy_for_single_layer.emplace_back((*lstm_cell_out)[1]);
      std::reverse(final_hy_for_single_layer.begin(), final_hy_for_single_layer.end());
      std::reverse(final_cy_for_single_layer.begin(), final_cy_for_single_layer.end());
      final_hy.emplace_back(JUST(functional::Concat(final_hy_for_single_layer, 0)));
      final_cy.emplace_back(JUST(functional::Concat(final_cy_for_single_layer, 0)));

      if (dropout != 0 && train && l < num_layers - 1) {
        std::shared_ptr<one::Tensor> stack_res = JUST(functional::Concat(*rnn_inputs, 0));
        std::shared_ptr<one::Tensor> dropout_res =
            JUST(functional::Dropout(stack_res, dropout, train, false, generator, nullptr));
        int64_t input_offset = 0;
        for (int32_t i = 0; i < num_steps; ++i) {
          const int64_t batch_size = batch_sizes_vec[i];
          (*rnn_inputs)[i] = JUST(functional::Narrow(dropout_res, 0, input_offset, batch_size));
          input_offset += batch_size;
        }
      }
    }
  }

  TensorTuple output;
  std::shared_ptr<one::Tensor> output_0 = JUST(functional::Concat(*rnn_inputs, 0));
  output.emplace_back(output_0);
  output.emplace_back(JUST(functional::Stack(final_hy, 0)));
  output.emplace_back(JUST(functional::Stack(final_cy, 0)));
  return output;
}

class LstmDataFunctor {
 public:
  LstmDataFunctor() {}

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& data,
                                const std::shared_ptr<one::Tensor>& batch_sizes,
                                const one::TensorTuple& hx, const one::TensorTuple& params,
                                const bool& has_biases, const int32_t& num_layers,
                                const float& dropout, const bool& train,
                                const bool& bidirectional) const {
    return _lstm_pack_sequence_impl(data, batch_sizes, hx, params, has_biases, num_layers, dropout,
                                    train, bidirectional);
  }
};

class GruInputFunctor {
 public:
  GruInputFunctor() {}

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& input,
                                const std::shared_ptr<one::Tensor>& hx,
                                const one::TensorTuple& params, const bool& has_biases,
                                const int32_t& num_layers, const float& dropout, const bool& train,
                                const bool& bidirectional, const bool& batch_first) const {
    return _rnn_impl<GRUCell<CellParams>>(input, hx, params, has_biases, num_layers, dropout, train,
                                          bidirectional, batch_first);
  }
};

class GruDataFunctor {
 public:
  GruDataFunctor() {}

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& data,
                                const std::shared_ptr<one::Tensor>& batch_sizes,
                                const std::shared_ptr<one::Tensor>& hx,
                                const one::TensorTuple& params, const bool& has_biases,
                                const int32_t& num_layers, const float& dropout, const bool& train,
                                const bool& bidirectional) const {
    return _rnn_pack_sequence_impl<GRUCell<CellParams>>(data, batch_sizes, hx, params, has_biases,
                                                        num_layers, dropout, train, bidirectional);
  }
};

Maybe<void> checkLongTensor(const std::shared_ptr<one::Tensor>& tensor) {
  auto& device = JUST(tensor->device())->type();
  CHECK_OR_RETURN(tensor->ndim() == 1 && device == "cpu" && tensor->dtype() == DType::Int64())
      << "'lengths' argument should be a 1D CPU int64 tensor, but got " << tensor->ndim() << "D "
      << device << " " << tensor->dtype()->name() << " tensor";
  return Maybe<void>::Ok();
}

class PackPaddedSequenceFunctor {
 public:
  PackPaddedSequenceFunctor() {}

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& input,
                                const std::shared_ptr<one::Tensor>& lengths,
                                const bool& batch_first) const {
    CHECK_OR_RETURN(input->is_local() && lengths->is_local())
        << "pack_padded_sequence only accept local tensors as input.";
    std::shared_ptr<one::Tensor> new_input = input;
    if (batch_first) {
      std::vector<int32_t> dims;
      dims.resize(input->shape()->NumAxes());
      dims[0] = 1;
      dims[1] = 0;
      for (int i = 2; i < input->shape()->NumAxes(); ++i) { dims[i] = i; }
      new_input = JUST(functional::Permute(input, dims));
    }
    JUST(checkLongTensor(lengths));

    int64_t batch_size = new_input->shape()->At(1);
    std::vector<int64_t> lengths_vec;
    lengths_vec.resize(lengths->nelement());
    const auto& callback = [&](ep::Stream* stream,
                               const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
      SyncAutoMemcpy(stream, lengths_vec.data(), eager_blob_object->dptr(),
                     lengths_vec.size() * sizeof(int64_t), memory::MakeHostMemCase(),
                     eager_blob_object->mem_case());
    };
    JUST(SyncAccessTensorWithTimeOut(lengths, callback, "const"));

    CHECK_OR_RETURN(new_input->nelement() > 0) << "Cannot pack empty tensors.";
    CHECK_OR_RETURN(lengths->shape()->At(0) == batch_size)
        << "Expected `len(lengths)` to be equal to batch_size, but got " << lengths->shape()->At(0)
        << " (batch_size=" << batch_size << ")";
    CHECK_OR_RETURN(lengths_vec[batch_size - 1] > 0)
        << "Length of all samples has to be greater than 0, but found an element in 'lengths' that "
           "is <= 0";
    for (int i = 0; i < batch_size - 1; ++i) {
      if (lengths_vec[batch_size - 1 - i] > lengths_vec[batch_size - 2 - i]) {
        CHECK_OR_RETURN(false) << "`lengths` array must be sorted in decreasing order when "
                                  "`enforce_sorted` is True. You can pass `enforce_sorted=False` "
                                  "to pack_padded_sequence and/or pack_sequence to sidestep this "
                                  "requirement if you do not need ONNX exportability.";
      }
    }

    std::vector<int64_t> step_shape_vec;  // == [-1, *input.shape[2:]]
    {
      const auto& input_sizes = new_input->shape();
      step_shape_vec.push_back(-1);
      for (int i = 2; i < input_sizes->NumAxes(); ++i) {
        step_shape_vec.push_back(input_sizes->At(i));
      }
    }
    DimVector rsv(step_shape_vec.size());
    for (int i = 0; i < step_shape_vec.size(); ++i) { rsv[i] = step_shape_vec[i]; }
    const Shape step_shape(rsv);

    // To understand what's going on in this loop imagine that the input is a padded 2D
    // array that looks like this (x = valid entry, . = padding)
    //
    //  1 1 1 1 1
    //  2 2 2 . .
    //  2 2 2 . .
    //  4 . . . .
    //  4 . . . .
    //
    // Where the vertical dimension corresponds to time, and horizontal dim to batch.
    // In this example, the lengths array will be equal to [5, 3, 3, 1, 1], and we will
    // iterate over them in reverse order (from the rightmost column to the left).
    // We want to avoid eager slicing of the input at every time step, and wait for
    // the moments where the length increases. In this example, that will happen at the
    // first, second and fourth steps. Then, we slice out the whole block of the input
    // that corresponds to this length, and hasn't been sliced yet (the steps at which each
    // element is sliced are annotated in the array above).  You can think of this as if we
    // were scanning the sequences from the shortest one, and every time we realize there's
    // more elements below in our column, we lower the counter (prev_l), and append the new
    // block to the output.
    std::vector<int64_t> batch_sizes;
    batch_sizes.resize(lengths_vec[0]);
    int64_t* batch_sizes_ptr = batch_sizes.data();
    TensorTuple steps;
    int64_t prev_l = 0;
    for (int i = 0; i < batch_size; ++i) {
      int64_t l = lengths_vec[batch_size - 1 - i];
      if (l > prev_l) {
        auto current_batch_size = batch_size - i;
        std::shared_ptr<Tensor> slice_res =
            JUST(functional::Narrow(new_input, 0, prev_l, l - prev_l));
        slice_res = JUST(functional::Narrow(slice_res, 1, 0, current_batch_size));
        slice_res = JUST(functional::View(slice_res->contiguous(), step_shape));
        steps.emplace_back(slice_res);
        for (int64_t j = 0; j < (l - prev_l); ++j) { (*batch_sizes_ptr++) = current_batch_size; }
        prev_l = l;
      }
      CHECK_OR_RETURN(l >= prev_l)
          << "PackPaddedSequenceFunctor: `lengths` array must be sorted in decreasing order.";
    }

    DimVector lsv(1);
    lsv[0] = lengths_vec[0];
    const Shape ls(lsv);
    std::shared_ptr<Tensor> batch_sizes_t =
        JUST(functional::Empty(ls, lengths->dtype(), JUST(lengths->device()),
                               /*requires_grad=*/lengths->requires_grad(), /*pin_memory=*/false));
    const auto& callback2 = [&](ep::Stream* stream,
                                const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
      SyncAutoMemcpy(stream, eager_blob_object->mut_dptr(), batch_sizes.data(),
                     batch_sizes.size() * sizeof(int64_t), eager_blob_object->mem_case(),
                     memory::MakeHostMemCase());  // copy 1 scalar(int64_t) tensor's value to max
    };
    JUST(SyncAccessTensorWithTimeOut(batch_sizes_t, callback2, "const"));

    std::shared_ptr<TensorTuple> output = std::make_shared<TensorTuple>(2);
    (*output)[0] = JUST(functional::Concat(steps, 0));
    (*output)[1] = batch_sizes_t;
    return output;
  }
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::RnnTanhCellFunctor>("RnnTanhCell");
  m.add_functor<impl::RnnReluCellFunctor>("RnnReluCell");
  m.add_functor<impl::LstmCellFunctor>("LstmCell");
  m.add_functor<impl::GruCellFunctor>("GruCell");
  m.add_functor<impl::FusedLstmCellFunctor>("FusedLstmCell");
  m.add_functor<impl::FusedLstmCellGradFunctor>("FusedLstmCellGrad");
  m.add_functor<impl::FusedGruCellFunctor>("FusedGruCell");
  m.add_functor<impl::FusedGruCellGradFunctor>("FusedGruCellGrad");
  m.add_functor<impl::RnnTanhInputFunctor>("RnnTanhInput");
  m.add_functor<impl::RnnTanhDataFunctor>("RnnTanhData");
  m.add_functor<impl::RnnReluInputFunctor>("RnnReluInput");
  m.add_functor<impl::RnnReluDataFunctor>("RnnReluData");
  m.add_functor<impl::LstmInputFunctor>("LstmInput");
  m.add_functor<impl::LstmDataFunctor>("LstmData");
  m.add_functor<impl::GruInputFunctor>("GruInput");
  m.add_functor<impl::GruDataFunctor>("GruData");
  m.add_functor<impl::PackPaddedSequenceFunctor>("PackPaddedSequence");
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
