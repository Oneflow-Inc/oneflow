#include "oneflow/core/kernel/yolo_prob_loss_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void CalcObjnessDiffGpu(const int32_t* valid_num_ptr, const int32_t* inds_ptr,
                                   const T* bbox_objness_ptr, T* bbox_objness_out_ptr,
                                   int32_t value) {
  size_t num;
  if (value == 1) {
    num = valid_num_ptr[0];
  } else {
    num = valid_num_ptr[1];
  }
  CUDA_1D_KERNEL_LOOP(i, num) {
    int32_t box_index = inds_ptr[i];
    bbox_objness_out_ptr[box_index] = bbox_objness_ptr[box_index] - value;
  }
}

template<typename T>
__global__ void CopyValidClsProbGpu(const int32_t* valid_num_ptr, const int32_t num_clsprobs,
                                    const int32_t* pos_inds_ptr, const T* bbox_clsprob_ptr,
                                    T* bbox_clsprob_out_ptr) {
  CUDA_1D_KERNEL_LOOP(index, valid_num_ptr[0] * num_clsprobs) {
    size_t i = index / num_clsprobs;
    size_t j = index % num_clsprobs;
    int32_t box_index = pos_inds_ptr[i];
    (bbox_clsprob_out_ptr + num_clsprobs * box_index)[j] =
        (bbox_clsprob_ptr + num_clsprobs * box_index)[j];
  }
}

template<typename T>
__global__ void CalcClsProbDiffGpu(const int32_t* valid_num_ptr, const int32_t num_clsprobs,
                                   const int32_t* pos_inds_ptr, const int32_t* pos_cls_label_ptr,
                                   T* bbox_clsprob_out_ptr) {
  CUDA_1D_KERNEL_LOOP(i, valid_num_ptr[0]) {
    int32_t box_index = pos_inds_ptr[i];
    if (pos_cls_label_ptr[box_index] >= 0) {
      int32_t idx = num_clsprobs * box_index + pos_cls_label_ptr[box_index];
      bbox_clsprob_out_ptr[idx]--;
    }
  }
}

}  // namespace

template<typename T>
struct YoloProbLossKernelUtil<DeviceType::kGPU, T> {
  static void CalcObjnessDiff(DeviceCtx* ctx, const size_t pos_num, const size_t neg_num,
                              const int32_t* valid_num_ptr, const int32_t* pos_inds_ptr,
                              const int32_t* neg_inds_ptr, const T* bbox_objness_ptr,
                              T* bbox_objness_out_ptr) {
    CalcObjnessDiffGpu<T>
        <<<BlocksNum4ThreadsNum(pos_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            valid_num_ptr, pos_inds_ptr, bbox_objness_ptr, bbox_objness_out_ptr, 1);
    CalcObjnessDiffGpu<T>
        <<<BlocksNum4ThreadsNum(neg_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            valid_num_ptr, neg_inds_ptr, bbox_objness_ptr, bbox_objness_out_ptr, 0);
  }
  static void CalcClsProbDiff(DeviceCtx* ctx, const size_t pos_num, const int32_t num_clsprobs,
                              const int32_t* valid_num_ptr, const int32_t* pos_inds_ptr,
                              const int32_t* pos_cls_label_ptr, const T* bbox_clsprob_ptr,
                              T* bbox_clsprob_out_ptr) {
    CopyValidClsProbGpu<T><<<BlocksNum4ThreadsNum(pos_num * num_clsprobs), kCudaThreadsNumPerBlock,
                             0, ctx->cuda_stream()>>>(valid_num_ptr, num_clsprobs, pos_inds_ptr,
                                                      bbox_clsprob_ptr, bbox_clsprob_out_ptr);
    // CudaCheck(cudaStreamSynchronize(ctx->cuda_stream()));
    CalcClsProbDiffGpu<T>
        <<<BlocksNum4ThreadsNum(pos_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            valid_num_ptr, num_clsprobs, pos_inds_ptr, pos_cls_label_ptr, bbox_clsprob_out_ptr);
    // CudaCheck(cudaStreamSynchronize(ctx->cuda_stream()));
  }
};  // namespace oneflow

#define INSTANTIATE_YOLO_PROB_LOSS_KERNEL_UTIL(type_cpp, type_proto) \
  template struct YoloProbLossKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_YOLO_PROB_LOSS_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
