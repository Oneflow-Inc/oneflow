#ifndef ONEFLOW_CORE_KERNEL_YOLO_KERNEL_UTIL_CU_H_
#define ONEFLOW_CORE_KERNEL_YOLO_KERNEL_UTIL_CU_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

size_t InferTempStorageForCUBYoloDetect(int box_num, int probs_num, int prob_thresh);

size_t InferTempStorageForCUBYoloBoxDiff(int box_num);

cudaError_t SelectOutIndexes(cudaStream_t stream, const float* probs_ptr, char* temp_storage_ptr,
                             int32_t* out_inds_ptr, int32_t* valid_num_ptr,
                             size_t temp_storage_bytes, int32_t box_num, int probs_num,
                             float prob_thresh);

cudaError_t SelectSamples(cudaStream_t stream, const int32_t* max_overlaps_gt_indices_ptr,
                          char* temp_storage_ptr, int32_t* pos_inds_ptr, int32_t* neg_inds_ptr,
                          int32_t* valid_num_ptr, size_t temp_storage_bytes, int32_t box_num);
}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_YOLO_KERNEL_UTIL_CU_H_
