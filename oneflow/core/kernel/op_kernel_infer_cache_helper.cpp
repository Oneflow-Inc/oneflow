#include "oneflow/core/kernel/op_kernel_infer_cache_helper.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/common/cached_caller.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace user_op {

OpKernelInferCacheHelper::OpKernelInferCacheHelper(const KernelConf& kernel_conf,
                                                   const JobDesc& job_desc) {
  const OperatorConf& op_conf = kernel_conf.op_attribute().op_conf();
  std::shared_ptr<Operator> op = ConstructOp(op_conf, &job_desc);
  cache_key_.job_desc = &job_desc;
  cache_key_.op_conf_sym = op->GetOpConfWithoutOpNameAndLbn();
  cache_key_.ibn_idx2shape_sym.resize(op->input_bns().size());
  cache_key_.dtype_signature_sym = SymbolOf(kernel_conf.dtype_signature());
}

void OpKernelInferCacheHelper::ForwardShape(std::function<void(KernelInferContext*)> infer_fn,
                                            KernelInferContext* ctx) {
  UpdateCacheKey(ctx);
  auto Infer = [=](const OpInferCacheKey& key) -> std::shared_ptr<const OpInferCacheValue> {
    infer_fn(ctx);
    auto* cache_value = new OpInferCacheValue();
    cache_value->obn_idx2shape_sym.resize(ctx->outputs().size());
    FOR_RANGE(int, i, 0, ctx->outputs().size()) {
      const auto& out_arg_pair = ctx->outputs().at(i);
      const ShapeView& out_shape_view =
          ctx->ShapeView4ArgNameAndIndex(out_arg_pair.first, out_arg_pair.second);
      Shape out_shape;
      out_shape_view.ToShape(&out_shape);
      cache_value->obn_idx2shape_sym.at(i).reset(out_shape);
    }
    return std::shared_ptr<const OpInferCacheValue>(cache_value);
  };
  size_t cache_size = Global<ResourceDesc>::Get()->thread_local_cache_max_size();
  auto cache_value_ptr = ThreadLocalCachedCall(cache_size, Infer, cache_key_);
  FOR_RANGE(int, i, 0, ctx->outputs().size()) {
    const auto& out_arg_pair = ctx->outputs().at(i);
    auto* mut_shape_view =
        ctx->MutShapeView4ArgNameAndIndex(out_arg_pair.first, out_arg_pair.second);
    mut_shape_view->set_shape(*cache_value_ptr->obn_idx2shape_sym.at(i));
  }
}

void OpKernelInferCacheHelper::UpdateCacheKey(KernelInferContext* ctx) {
  auto GetSymbolOfShape = [=](const std::string& arg_name, int32_t arg_index) -> Symbol<Shape> {
    Shape shape;
    shape.LeftOnesExtendedAssign(ctx->ShapeView4ArgNameAndIndex(arg_name, arg_index));
    return SymbolOf(shape);
  };
  const auto& inputs = ctx->inputs();
  FOR_RANGE(int, i, 0, inputs.size()) {
    const auto& arg_pair = inputs.at(i);
    cache_key_.ibn_idx2shape_sym.at(i) = GetSymbolOfShape(arg_pair.first, arg_pair.second);
  }
}

}  // namespace user_op

}  // namespace oneflow
