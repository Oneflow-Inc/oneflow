#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include <math.h>
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"

namespace oneflow {

namespace user_op {

#define REDUCE_GPU_SEQ               \
  OF_PP_MAKE_TUPLE_SEQ("Prod", Prod) \
  OF_PP_MAKE_TUPLE_SEQ("Any", Any)   \
  OF_PP_MAKE_TUPLE_SEQ("Min", Min)

template<typename T>
class ReduceGpuKernel final : public OpKernel {
 public:
  ReduceGpuKernel() = default;
  ~ReduceGpuKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* tensor_in = ctx->Tensor4ArgNameAndIndex("tensor_in", 0);
    Tensor* tensor_out = ctx->Tensor4ArgNameAndIndex("tensor_out", 0);
    Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    auto axis = ctx->GetAttr<std::vector<int32_t>>("axis");
    const Shape& reduced_shape =
        axis.empty() ? Shape::Ones(tensor_in->shape().NumAxes())
                     : CreateReducedShape(tensor_in->shape(), {axis.begin(), axis.end()});
    std::string reduce_func_type = ctx->GetAttr<std::string>("reduce_func_type");

#define REDUCE_FORWARD(reduce_func_type_str, func_name_postfix)                        \
  if (reduce_func_type == reduce_func_type_str) {                                      \
    NdarrayUtil<DeviceType::kGPU, T>::Reduce##func_name_postfix(                       \
        ctx->device_ctx(), XpuVarNdarray<T>(reduced_shape, tensor_out->mut_dptr<T>()), \
        XpuVarNdarray<const T>(tensor_in->shape(), tensor_in->dptr<T>()),              \
        XpuVarNdarray<T>(tmp_buffer->shape(), tmp_buffer->mut_dptr<T>()));             \
  }
    OF_PP_FOR_EACH_TUPLE(REDUCE_FORWARD, REDUCE_GPU_SEQ);
#undef REDUCE_FORWARD
  }
};

#define REGISTER_REDUCE_GPU_KERNEL(dtype)                                                    \
  REGISTER_USER_KERNEL("reduce")                                                             \
      .SetCreateFn<ReduceGpuKernel<dtype>>()                                                 \
      .SetIsMatchedPred([](const KernelRegContext& ctx) {                                    \
        const TensorDesc* tensor_out_desc = ctx.TensorDesc4ArgNameAndIndex("tensor_out", 0); \
        if (ctx.device_type() == DeviceType::kGPU                                            \
            && tensor_out_desc->data_type() == GetDataType<dtype>::value) {                  \
          return true;                                                                       \
        }                                                                                    \
        return false;                                                                        \
      })                                                                                     \
      .SetInferTmpSizeFn([](InferContext* ctx) {                                             \
        const Shape* in_shape = ctx->Shape4ArgNameAndIndex("tensor_in", 0);                  \
        return in_shape->elem_cnt() * sizeof(dtype);                                         \
      });

REGISTER_REDUCE_GPU_KERNEL(float)
REGISTER_REDUCE_GPU_KERNEL(double)
REGISTER_REDUCE_GPU_KERNEL(int32_t)
REGISTER_REDUCE_GPU_KERNEL(int64_t)
}  // namespace user_op
}  // namespace oneflow