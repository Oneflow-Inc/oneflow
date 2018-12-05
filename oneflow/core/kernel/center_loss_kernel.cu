#include "oneflow/core/kernel/center_loss_kernel.h"

namespace oneflow {

namespace {

template<typename PredType, typename LabelType>
__global__ void LookupGpu(const int64_t elem_cnt, const LabelType* indices, const PredType* in,
                          int32_t in_row_num, int32_t in_col_num, PredType* out) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const int32_t out_idx = i / in_col_num;
    const int32_t offset = i % in_col_num;
    const int32_t idx = indices[out_idx];
    assert(idx >= 0 && idx < in_row_num);
    out[i] = in[idx * in_col_num + offset];
  }
}

template<typename PredType, typename LabelType>
__global__ void SparseUpdateGpu(const int64_t elem_cnt, const LabelType* indices,
                                const PredType* diff, int32_t diff_row_num, int32_t diff_col_num,
                                PredType* model, const int32_t model_row_num) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const int32_t model_row_idx = indices[i / diff_col_num];
    const int32_t model_offset = i % diff_col_num;
    assert(model_row_idx >= 0 && model_row_idx < model_row_num);
    model[model_row_idx * diff_col_num + model_offset] -= diff[i];
  }
}

}  // namespace

template<typename PredType, typename LabelType>
struct CenterLossKernelUtil<DeviceType::kGPU, PredType, LabelType> {
  static void Lookup(DeviceCtx* ctx, const PredType* in, const int32_t in_row_num,
                     const int32_t in_col_num, const LabelType* indices,
                     const int32_t num_of_indices, PredType* out) {
    const int64_t elem_cnt = num_of_indices * in_col_num;
    LookupGpu<PredType, LabelType>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, indices, in, in_row_num, in_col_num, out);
  }
  static void SparseUpdate(DeviceCtx* ctx, const PredType* diff, const int32_t diff_row_num,
                           const int32_t diff_col_num, const LabelType* indices,
                           int32_t num_of_indices, PredType* model, const int32_t model_row_num) {
    const int64_t elem_cnt = num_of_indices * diff_col_num;
    SparseUpdateGpu<PredType, LabelType>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, indices, diff, diff_row_num, diff_col_num, model, model_row_num);
  }
};

#define MAKE_ENTRY(data_type_pair, label_type_pair)                                        \
  template struct CenterLossKernelUtil<DeviceType::kGPU, OF_PP_PAIR_FIRST(data_type_pair), \
                                       OF_PP_PAIR_FIRST(label_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)

}  // namespace oneflow