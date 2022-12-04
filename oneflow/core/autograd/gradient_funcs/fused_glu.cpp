#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct FusedGluGradCaptureState : public AutoGradCaptureState {
    bool is_split_mode = false;
    std::string activation = "none";
    bool w_requires_grad = false;
    bool v_requires_grad = false;
    bool b_requires_grad = false;
    bool c_requires_grad = false;
};

class FusedGluGrad : public OpExprGradFunction<FusedGluGradCaptureState>{
    Maybe<void> Init(const OpExpr& op) override;

    Maybe<void> Capture(FusedGluGradCaptureState* ctx, const TensorTuple& inputs,
                        const TensorTuple& outputs, const AttrMap& attrs) const override;

    Maybe<void> Apply(const FusedGluGradCaptureState* ctx, const TensorTuple& out_grads,
                        TensorTuple* in_grads) const override;
private:
    AttrMap base_attrs_;
};

Maybe<void> FusedGluGrad::Init(const OpExpr& op){
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
}

Maybe<void> FusedGluGrad::Capture(FusedGluGradCaptureState* ctx, const TensorTuple& inputs,
                        const TensorTuple& outputs, const AttrMap& attrs) const {
    // check input size
    const size_t in_size = inputs.size();
    CHECK_OR_RETURN(in_size == 3 || in_size == 5)
        << "FusedGluGrad::Capture(): input tensor size must be 3 or 5";
    if(in_size == 5){
        ctx->is_split_mode = true;
    }
    
    // check whether input tensors need grad
    ctx->w_requires_grad = inputs[1]->requires_grad();
    ctx->b_requires_grad = inputs[2]->requires_grad();
    if(ctx->is_split_mode) {
        ctx->v_requires_grad = inputs[3]->requires_grad();
        ctx->c_requires_grad = inputs[4]->requires_grad();
    }

    // save tensors for backward
    ctx->SaveTensorForBackward(inputs[0]);   // x
    ctx->SaveTensorForBackward(outputs[1]);  // matmul_wx
    if(ctx->is_split_mode) {
        ctx->SaveTensorForBackward(outputs[2]);  // matmul_vx
    }

    // save activation type
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->activation = JUST(composed_attrs.GetAttr<std::string>("activation"));

    return Maybe<void>::Ok();
}

Maybe<void> FusedGluGrad::Apply(const FusedGluGradCaptureState* ctx, const TensorTuple& out_grads,
                        TensorTuple* in_grads) const {
    // obtain saved tensors from forward process
    const auto& x = ctx->SavedTensors()[0];
    const auto& matmul_wx = ctx->SavedTensors()[1];

    // obtain gradient dy
    const auto& dy = out_grads[0];

    if(ctx->is_split_mode){
        // obtain saved optional tensor from forward process
        const auto& matmul_vx = ctx->SavedTensors()[2];

        // calculate the intermediate gradient using fused kernel
        const auto& middle_results = JUST(functional::FusedGluWithoutLinearGrad(dy, matmul_wx, matmul_vx, ctx->activation));
        const auto& d_matmul_wx = (*middle_results)[0];
        const auto& d_matmul_vx = (*middle_results)[1];

        // calculate the final result
        const auto& wb_results = JUST(functional::CublasMatmulBiasAddGrad(d_matmul_wx, x));
        const auto& vc_results = JUST(functional::CublasMatmulBiasAddGrad(d_matmul_vx, x));
        const auto& w_grad = (*wb_results)[0];
        const auto& b_grad = (*wb_results)[1];
        const auto& v_grad = (*vc_results)[0];
        const auto& c_grad = (*vc_results)[1];

        // update gradients
        if(ctx->w_requires_grad){ (*in_grads)[1] = w_grad; }
        if(ctx->b_requires_grad){ (*in_grads)[2] = b_grad; }
        if(ctx->v_requires_grad){ (*in_grads)[3] = v_grad; }
        if(ctx->c_requires_grad){ (*in_grads)[4] = c_grad; }
    } else {
        // calculate the intermediate gradient using fused kernel
        const auto& middle_results = JUST(functional::FusedGluWithoutLinearGrad(dy, matmul_wx, nullptr, ctx->activation));
        const auto& d_matmul_wx = (*middle_results)[0];

        // calculate the final result
        const auto& results = JUST(functional::CublasMatmulBiasAddGrad(d_matmul_wx, x));
        const auto& w_grad = (*results)[0];
        const auto& b_grad = (*results)[1];

        // update gradients
        if(ctx->w_requires_grad){ (*in_grads)[1] = w_grad; }
        if(ctx->b_requires_grad){ (*in_grads)[2] = b_grad; }
    }

    return Maybe<void>::Ok();
}
    
}
}