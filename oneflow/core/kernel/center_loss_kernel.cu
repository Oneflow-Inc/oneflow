#include "oneflow/core/kernel/center_loss_kernel.h"

namespace oneflow {

namespace {

template<typename PredType, typename LabelType>
__global__ void LookupGpu(const int64_t elem_cnt, const LabelType* indices, const PredType* in,
                          int32_t table_size, int32_t table_dim, PredType* out) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const int32_t out_idx = i / table_dim;
    const int32_t out_offset = i % table_dim;
    const int32_t idx = indices[out_idx];
    assert(idx >= 0 && idx < table_size);
    out[i] = in[idx * table_dim + out_offset];
  }
}

template<typename PredType, typename LabelType>
__global__ void CauculateEuclideanDistanceGpu(const int64_t elem_cnt, const PredType* x,
                                              const PredType* y, PredType* z) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    PredType diff = x[i] - y[i];
    z[i] = 0.5 * diff * diff;
  }
}

template<typename PredType, typename LabelType>
__global__ void SparseUpdateGpu(const int64_t elem_cnt, const LabelType* indices,
                                const PredType* diff, int32_t update_num, int32_t table_dim,
                                PredType* model, const int32_t table_size) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const int32_t model_row_idx = indices[i / table_dim];
    const int32_t model_offset = i % table_dim;
    assert(model_row_idx >= 0 && model_row_idx < table_size);
    model[model_row_idx * table_dim + model_offset] -= diff[i];
  }
}

}  // namespace

template<typename PredType, typename LabelType>
struct CenterLossKernelUtil<DeviceType::kGPU, PredType, LabelType> {
  static void Lookup(DeviceCtx* ctx, const PredType* in, const int32_t table_size,
                     const int32_t table_dim, const LabelType* indices,
                     const int32_t num_of_indices, PredType* out) {
    const int64_t elem_cnt = num_of_indices * table_dim;
    LookupGpu<PredType, LabelType>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, indices, in, table_size, table_dim, out);
  }
  static void CalculateEuclideanDistance(DeviceCtx* ctx, const int64_t elem_cnt, const PredType* x,
                                         const PredType* y, PredType* z) {
    CauculateEuclideanDistanceGpu<PredType, LabelType>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, x, y, z);
  }
  static void SparseUpdate(DeviceCtx* ctx, const PredType* diff, const int32_t update_num,
                           const int32_t table_dim, const LabelType* indices, PredType* model,
                           const int32_t table_size) {
    const int64_t elem_cnt = update_num * table_dim;
    SparseUpdateGpu<PredType, LabelType>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, indices, diff, update_num, table_dim, model, table_size);
  }
};

#define MAKE_ENTRY(data_type_pair, label_type_pair)                                        \
  template struct CenterLossKernelUtil<DeviceType::kGPU, OF_PP_PAIR_FIRST(data_type_pair), \
                                       OF_PP_PAIR_FIRST(label_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)

}  // namespace oneflow
