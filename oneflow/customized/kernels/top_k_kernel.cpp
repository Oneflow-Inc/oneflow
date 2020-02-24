#include "oneflow/core/framework/framework.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

namespace {

template<typename T>
void ForwardPartDataContentTopOne(const T* in_ptr, const Range& range, const int32_t instance_size,
                                  int32_t* out_ptr) {
  FOR_RANGE(int32_t, i, range.begin(), range.end()) {
    const T* in_ptr_i = in_ptr + i * instance_size;
    out_ptr[i] = std::distance(in_ptr_i, std::max_element(in_ptr_i, in_ptr_i + instance_size));
  }
}

template<typename T>
void ForwardPartDataContentTopK(const T* in_ptr, int32_t* indices_ptr, const Range& range,
                                const int32_t instance_size, const int32_t k, const bool sorted,
                                int32_t* out_ptr) {
  CHECK_NOTNULL(indices_ptr);
  FOR_RANGE(int32_t, i, range.begin(), range.end()) {
    const int32_t offset = i * instance_size;
    int32_t* indices_ptr_i = indices_ptr + offset;
    const T* in_ptr_i = in_ptr + offset;
    std::iota(indices_ptr_i, indices_ptr_i + instance_size, 0);
    auto comp = [&](const int32_t lhs, const int32_t rhs) {
      const T l = in_ptr_i[lhs];
      const T r = in_ptr_i[rhs];
      if (l == r) {
        return lhs < rhs;
      } else {
        return l > r;
      }
    };
    std::nth_element(indices_ptr_i, indices_ptr_i + k, indices_ptr_i + instance_size, comp);
    if (sorted) { std::sort(indices_ptr_i, indices_ptr_i + k, comp); }
    std::copy(indices_ptr_i, indices_ptr_i + k, out_ptr + i * k);
  }
}

template<typename T>
void CpuTopK(DeviceCtx* ctx, const T* in_ptr, int32_t* indices_ptr, int32_t instance_num,
             int32_t instance_size, int32_t k, bool sorted, int32_t* out_ptr) {
  const int32_t part_num =
      std::min(instance_num, Global<ThreadMgr>::Get()->compute_thread_pool()->thread_num());
  const BalancedSplitter bs(instance_num, part_num);
  BlockingCounter bc(part_num);
  FOR_RANGE(int32_t, part_id, 0, part_num) {
    const Range range = bs.At(part_id);
    Global<ThreadMgr>::Get()->compute_thread_pool()->AddWork([=, &bc]() {
      if (k == 1) {
        ForwardPartDataContentTopOne(in_ptr, range, instance_size, out_ptr);
      } else {
        ForwardPartDataContentTopK(in_ptr, indices_ptr, range, instance_size, k, sorted, out_ptr);
      }
      bc.Decrease();
    });
  }
  bc.WaitUntilCntEqualZero();
}

}  // namespace

template<typename T>
class TopKCpuKernel final : public user_op::OpKernel {
 public:
  TopKCpuKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  TopKCpuKernel() = default;
  ~TopKCpuKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const int32_t instance_size = in->shape().At(in->shape().NumAxes() - 1);
    const int32_t instance_num = in->shape().elem_cnt() / instance_size;
    const int32_t k = std::min(ctx->GetAttr<int32_t>("k"), instance_size);

    int32_t* indices_ptr = indices ? indices->mut_dptr<int32_t>() : nullptr;
    CpuTopK(ctx->device_ctx(), in->dptr<T>(), indices_ptr, instance_num, instance_size, k,
            ctx->GetAttr<bool>("sorted"), out->mut_dptr<int32_t>());
  };
};

#define REGISTER_TOP_K_CPU_KERNEL(dtype)                                               \
  REGISTER_USER_KERNEL("top_k")                                                        \
      .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) {                \
        return new TopKCpuKernel<dtype>(ctx);                                          \
      })                                                                               \
      .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) {            \
        return ctx.device() == DeviceType::kCPU                                        \
               && ctx.TensorDesc4ArgNameAndIndex("in", 0)->data_type()                 \
                      == GetDataType<dtype>::value;                                    \
      })                                                                               \
      .SetInferTmpSizeFn([](oneflow::user_op::InferContext* ctx) {                     \
        return ctx->GetAttr<int32_t>("k") > 1                                          \
                   ? ctx->Shape4ArgNameAndIndex("in", 0)->elem_cnt() * sizeof(int32_t) \
                   : 0;                                                                \
      });

REGISTER_TOP_K_CPU_KERNEL(float)
REGISTER_TOP_K_CPU_KERNEL(double)
REGISTER_TOP_K_CPU_KERNEL(int8_t)
REGISTER_TOP_K_CPU_KERNEL(int32_t)
REGISTER_TOP_K_CPU_KERNEL(int64_t)

}  // namespace oneflow