#include "oneflow/core/kernel/yolo_kernel_util.cuh"
#include <cub/cub.cuh>

namespace oneflow {

namespace {

struct IsObject {
  const float* probs_ptr;
  int probs_num;
  float prob_thresh;
  OF_DEVICE_FUNC
  IsObject(const float* probs_ptr, int probs_num, float prob_thresh)
      : probs_ptr(probs_ptr), probs_num(probs_num), prob_thresh(prob_thresh) {}
  OF_DEVICE_FUNC
  bool operator()(const int& i) const { return (probs_ptr[i * probs_num + 0] > prob_thresh); }
};

struct IsPostive {
  const int32_t* max_overlaps_gt_indices_ptr;
  OF_DEVICE_FUNC
  IsPostive(const int32_t* max_overlaps_gt_indices_ptr)
      : max_overlaps_gt_indices_ptr(max_overlaps_gt_indices_ptr) {}
  OF_DEVICE_FUNC
  bool operator()(const int& i) const { return (max_overlaps_gt_indices_ptr[i] >= 0); }
};

struct IsNegative {
  const int32_t* max_overlaps_gt_indices_ptr;
  OF_DEVICE_FUNC
  IsNegative(const int32_t* max_overlaps_gt_indices_ptr)
      : max_overlaps_gt_indices_ptr(max_overlaps_gt_indices_ptr) {}
  OF_DEVICE_FUNC
  bool operator()(const int& i) const { return (max_overlaps_gt_indices_ptr[i] == -1); }
};

}  // namespace

size_t InferTempStorageForCUBYoloDetect(int box_num, int probs_num, int prob_thresh) {
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  IsObject select_op(static_cast<float*>(NULL), probs_num, prob_thresh);
  auto err = cub::DeviceSelect::If(
      /* d_temp_storage */ d_temp_storage,
      /* temp_storage_bytes */ temp_storage_bytes,
      /* d_in */ static_cast<int*>(NULL),
      /* d_out */ static_cast<int*>(NULL),
      /* d_num_selected_out */ static_cast<int*>(NULL),
      /* num_items */ box_num,
      /* select_op */ select_op,
      /* stream */ nullptr,
      /* debug_synchronous */ false);
  CudaCheck(err);
  if (temp_storage_bytes == 0) { temp_storage_bytes = 1; }
  return temp_storage_bytes;
}

size_t InferTempStorageForCUBYoloBoxDiff(int box_num) {
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  IsPostive select_op(static_cast<int32_t*>(NULL));
  auto err = cub::DeviceSelect::If(
      /* d_temp_storage */ d_temp_storage,
      /* temp_storage_bytes */ temp_storage_bytes,
      /* d_in */ static_cast<int*>(NULL),
      /* d_out */ static_cast<int*>(NULL),
      /* d_num_selected_out */ static_cast<int*>(NULL),
      /* num_items */ box_num,
      /* select_op */ select_op,
      /* stream */ nullptr,
      /* debug_synchronous */ false);
  CudaCheck(err);
  if (temp_storage_bytes == 0) { temp_storage_bytes = 1; }
  return temp_storage_bytes;
}

cudaError_t SelectOutIndexes(cudaStream_t stream, const float* probs_ptr, char* temp_storage_ptr,
                             int32_t* out_inds_ptr, int32_t* valid_num_ptr,
                             size_t temp_storage_bytes, int32_t box_num, int probs_num,
                             float prob_thresh) {
  IsObject select_op(probs_ptr, probs_num, prob_thresh);
  cub::CountingInputIterator<int32_t> in_index_counter(0);

  auto err = cub::DeviceSelect::If(
      /* d_temp_storage */ temp_storage_ptr,
      /* temp_storage_bytes */ temp_storage_bytes,
      /* d_in */ in_index_counter,
      /* d_out */ out_inds_ptr,
      /* d_num_selected_out */ valid_num_ptr,
      /* num_items */ box_num,
      /* select_op */ select_op,
      /* stream */ stream,
      /* debug_synchronous */ false);
  return err;
}

cudaError_t SelectSamples(cudaStream_t stream, const int32_t* max_overlaps_gt_indices_ptr,
                          char* temp_storage_ptr, int32_t* pos_inds_ptr, int32_t* neg_inds_ptr,
                          int32_t* valid_num_ptr, size_t temp_storage_bytes, int32_t box_num) {
  IsPostive select_pos(max_overlaps_gt_indices_ptr);
  cub::CountingInputIterator<int32_t> pos_in_index_counter(0);
  auto err1 = cub::DeviceSelect::If(
      /* d_temp_storage */ temp_storage_ptr,
      /* temp_storage_bytes */ temp_storage_bytes,
      /* d_in */ pos_in_index_counter,
      /* d_out */ pos_inds_ptr,
      /* d_num_selected_out */ valid_num_ptr,
      /* num_items */ box_num,
      /* select_op */ select_pos,
      /* stream */ stream,
      /* debug_synchronous */ false);
  CudaCheck(err1);

  IsNegative select_neg(max_overlaps_gt_indices_ptr);
  cub::CountingInputIterator<int32_t> neg_in_index_counter(0);
  auto err2 = cub::DeviceSelect::If(
      /* d_temp_storage */ temp_storage_ptr,
      /* temp_storage_bytes */ temp_storage_bytes,
      /* d_in */ neg_in_index_counter,
      /* d_out */ neg_inds_ptr,
      /* d_num_selected_out */ valid_num_ptr + 1,
      /* num_items */ box_num,
      /* select_op */ select_neg,
      /* stream */ stream,
      /* debug_synchronous */ false);
  return err2;
}

}  // namespace oneflow
