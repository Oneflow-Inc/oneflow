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
#include "oneflow/core/common/error.pb.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#if CUDA_VERSION >= 11040

namespace oneflow {

namespace one {

struct CublasFusedMLPCaptureState : public AutoGradCaptureState {
    int32_t weight_num = 0; 
    bool skip_final_activation = false; 
};

class CublasFusedMLP : public OpExprGradFunction<CublasFusedMLPCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override;
  Maybe<void> Capture(CublasFusedMLPCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override;
  Maybe<void> Apply(const CublasFusedMLPCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 protected:
  AttrMap base_attrs_;
};

Maybe<void> CublasFusedMLP::Init(const OpExpr& op) {
  const UserOpExpr* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}

Maybe<void> CublasFusedMLP::Capture(CublasFusedMLPCaptureState* ctx,
                                            const TensorTuple& inputs, const TensorTuple& outputs,
                                            const AttrMap& attrs) const {
  int32_t weight_num = (inputs.size() - 1) / 2; 
  ctx->weight_num = weight_num; 
  
  ctx->SaveTensorForBackward(inputs.at(0)); // x. idx_sum:1
  for(int32_t i = 0; i < weight_num; i++){
    ctx->SaveTensorForBackward(inputs.at(i+1)); // weights. idx_sum:1+w
  }

  ctx->SaveTensorForBackward(outputs.at(0)); // final layers output. idx_sum:2+w
  for(int32_t i = 0; i < weight_num; i++){
    ctx->SaveTensorForBackward(outputs.at(i+1)); // cublas aux. need minus 1. idx_sum:2+2w 
  }
  for(int32_t i = 0; i < weight_num-1; i++){
    ctx->SaveTensorForBackward(outputs.at(i+1+weight_num)); // hidden. 
  }

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->skip_final_activation = JUST(composed_attrs.GetAttr<bool>("skip_final_activation"));
  return Maybe<void>::Ok();
}

Maybe<void> CublasFusedMLP::Apply(const CublasFusedMLPCaptureState* ctx,
                                          const TensorTuple& out_grads,
                                          TensorTuple* in_grads) const {
  int32_t weight_num = ctx->weight_num; 
  in_grads->resize(1+2*weight_num); 
  std::shared_ptr<one::Tensor> last_relu_grad; 
  if(!ctx->skip_final_activation){
    // step1: use dy and final output to get last layer's relu grad. 
    last_relu_grad = JUST(functional::ReluGrad(
      JUST(VectorAt(out_grads, 0)), JUST(VectorAt(ctx->SavedTensors(), 1+weight_num))));
  }else{
    last_relu_grad = JUST(VectorAt(out_grads, 0)); 
  }
  
  // step2: use reduce_sum to get last layer's bias grad. 
  std::vector<int32_t> reduce_axes_vec{0};
  in_grads->at(2*weight_num) = JUST(functional::ReduceSum(last_relu_grad, reduce_axes_vec, false));

  TensorTuple hiddens(weight_num-1); 
  TensorTuple weights(weight_num);
  TensorTuple cublas_auxs(weight_num);
  TensorTuple dgrad(weight_num); 
  
  std::shared_ptr<one::Tensor> x = ctx->SavedTensors().at(0);

  for(int32_t i = 0; i < weight_num; ++i) {
    weights[i] = ctx->SavedTensors().at(1+i); 
  }

  for(int32_t i = 0; i < weight_num; ++i) {
    cublas_auxs[i] = ctx->SavedTensors().at(i+2+weight_num); 
  }

  for(int32_t i = 0; i < weight_num-1; ++i) {
    hiddens[i] = ctx->SavedTensors().at(i+2+2*weight_num); 
  }
  
  for(int32_t hidden_layer_idx=weight_num-1; hidden_layer_idx>0; hidden_layer_idx--){
    if(hidden_layer_idx==weight_num-1){
        /*
        Here we use cublas to compute bias + relu + matmul grad. 
        Then use Matmul to compute weight grad. 
        if it is final layer, we use out_grads[0] as dy.
        */
        const auto& matmul_relu_bias_bgrad = JUST(functional::CublasBiasAddReluMatmulGrad(
                                                  last_relu_grad, weights.at(hidden_layer_idx), cublas_auxs.at(hidden_layer_idx-1)));
        // dgrad
        dgrad.at(hidden_layer_idx) = matmul_relu_bias_bgrad->at(0); 
        // dbias
        in_grads->at(weight_num+hidden_layer_idx) = matmul_relu_bias_bgrad->at(1); 
        // dw
        in_grads->at(1+hidden_layer_idx) = JUST(functional::MatMul(last_relu_grad, hiddens.at(hidden_layer_idx-1), true, false, 1.0));
    
    }else{
        const auto& matmul_relu_bias_bgrad = JUST(functional::CublasBiasAddReluMatmulGrad(
                                             dgrad.at(hidden_layer_idx+1), weights.at(hidden_layer_idx), cublas_auxs.at(hidden_layer_idx-1)));
        // dgrad
        dgrad.at(hidden_layer_idx) = matmul_relu_bias_bgrad->at(0);
        // dbias
        in_grads->at(weight_num+hidden_layer_idx) = matmul_relu_bias_bgrad->at(1);
        // dw
        in_grads->at(1+hidden_layer_idx) = JUST(functional::MatMul(dgrad.at(hidden_layer_idx+1), hiddens.at(hidden_layer_idx-1), true, false, 1.0));
    }
  }
  // For first layer, we need to use 2 matmul to get grads. 
  // dx: 
  in_grads->at(0) = JUST(functional::MatMul(dgrad.at(1), weights.at(0), false, false, 1.0));
  // dw: 
  in_grads->at(1) = JUST(functional::MatMul(dgrad.at(1), ctx->SavedTensors().at(0), true, false, 1.0));

  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("cublas_fused_mlp", CublasFusedMLP);

}  // namespace one

}  // namespace oneflow
#endif // CUDA_VERSION >= 11040
