#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

constexpr size_t kBlockSize = sizeof(int64_t) * 8;

template<typename T>
__host__ __device__ __forceinline__ T CeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

template<typename T>
__host__ __device__ __forceinline__ T IoU(T const* const a, T const* const b) {
  // T interS = max(min(a[2], b[2]) - max(a[0], b[0]) + 1, 0.f)
  //           * max(min(a[3], b[3]) - max(a[1], b[1]) + 1, 0.f);
  T interS = (min(a[0] + a[2] / 2, b[0] + b[2] / 2) - max(a[0] - a[2] / 2, b[0] - b[2] / 2))
             * (min(a[1] + a[3] / 2, b[1] + b[3] / 2) - max(a[1] - a[3] / 2, b[1] - b[3] / 2));
  T Sa = (a[2] * a[3]);
  T Sb = (b[2] * b[3]);
  return interS / (Sa + Sb - interS);
}

template<typename T>
__global__ void CalcSuppressionBitmaskMatrix(size_t num_boxes, const float nms_iou_threshold,
                                             const T* boxes, int64_t* suppression_bmask_matrix,
                                             const T* probs) {
  if (probs[0] == 0) { return; }
  const size_t row = blockIdx.y;
  const size_t col = blockIdx.x;

  const size_t row_size = min(num_boxes - row * kBlockSize, kBlockSize);
  const size_t col_size = min(num_boxes - col * kBlockSize, kBlockSize);

  __shared__ T block_boxes[kBlockSize * 4];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 4 + 0] = boxes[(kBlockSize * col + threadIdx.x) * 4 + 0];
    block_boxes[threadIdx.x * 4 + 1] = boxes[(kBlockSize * col + threadIdx.x) * 4 + 1];
    block_boxes[threadIdx.x * 4 + 2] = boxes[(kBlockSize * col + threadIdx.x) * 4 + 2];
    block_boxes[threadIdx.x * 4 + 3] = boxes[(kBlockSize * col + threadIdx.x) * 4 + 3];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const size_t cur_box_idx = kBlockSize * row + threadIdx.x;
    const T* cur_box_ptr = boxes + cur_box_idx * 4;
    // int i = 0;
    int64_t bits = 0;
    size_t start = 0;
    if (row == col) { start = threadIdx.x + 1; }
    for (size_t i = start; i < col_size; i++) {
      if (IoU(cur_box_ptr, block_boxes + i * 4) > nms_iou_threshold) { bits |= 1ll << i; }
    }
    const size_t col_blocks = CeilDiv(num_boxes, kBlockSize);
    suppression_bmask_matrix[cur_box_idx * col_blocks + col] = bits;
  }
}

template<typename T>
__global__ void ScanSuppression(const size_t num_boxes, const size_t num_blocks, size_t num_keep,
                                int64_t* suppression_bmask, int8_t* keep_mask, const T* probs) {
  if (probs[0] == 0) { return; }
  extern __shared__ int64_t remv[];
  remv[threadIdx.x] = 0;
  __syncthreads();
  for (size_t i = 0; i < num_boxes; ++i) {
    size_t block_n = i / kBlockSize;
    size_t block_i = i % kBlockSize;
    if (!(remv[block_n] & (1ll << block_i))) {
      remv[threadIdx.x] |= suppression_bmask[i * num_blocks + threadIdx.x];
      if (threadIdx.x == block_n && num_keep > 0) {
        keep_mask[i] = 1;
        num_keep -= 1;
      }
    }
    __syncthreads();
  }
}

}  // namespace

template<typename T>
class NmsGpuKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NmsGpuKernel);
  NmsGpuKernel() = default;
  ~NmsGpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const auto& op_conf = this->op_conf().non_maximum_suppression_conf();
    const Blob* bbox_blob = BnInOp2Blob("in");
    const T* bbox_ptr = bbox_blob->dptr<T>();
    const Blob* probs_blob = BnInOp2Blob("probs");
    const T* probs_ptr = probs_blob->dptr<T>();
    int64_t* suppression_ptr = BnInOp2Blob("fw_tmp")->mut_dptr<int64_t>();
    int8_t* keep_ptr = BnInOp2Blob("out")->mut_dptr<int8_t>();
    size_t num_boxes = bbox_blob->shape().At(1);  //(b, num_box, 4)
    size_t num_keep = num_boxes;
    if (op_conf.post_nms_top_n() > 0) {
      num_keep = std::min<size_t>(num_keep, op_conf.post_nms_top_n());
    }
    size_t num_blocks = CeilDiv(num_boxes, kBlockSize);
    Memset<DeviceType::kGPU>(ctx.device_ctx, suppression_ptr, 0,
                             bbox_blob->shape().At(0) * num_boxes * num_blocks * sizeof(int64_t));
    Memset<DeviceType::kGPU>(ctx.device_ctx, keep_ptr, 0,
                             bbox_blob->shape().At(0) * num_boxes * sizeof(int8_t));

    dim3 blocks(num_blocks, num_blocks);
    dim3 threads(kBlockSize);
    FOR_RANGE(int64_t, idx, 0, bbox_blob->shape().At(0)) {
      CalcSuppressionBitmaskMatrix<<<blocks, threads, 0, ctx.device_ctx->cuda_stream()>>>(
          num_boxes, op_conf.nms_iou_threshold(), bbox_ptr + idx * bbox_blob->shape().Count(1),
          suppression_ptr + idx * num_boxes * num_blocks,
          probs_ptr + idx * probs_blob->shape().Count(1));
      ScanSuppression<<<1, num_blocks, num_blocks, ctx.device_ctx->cuda_stream()>>>(
          num_boxes, num_blocks, num_keep, suppression_ptr + idx * num_boxes * num_blocks,
          keep_ptr + idx * num_boxes, probs_ptr + idx * probs_blob->shape().Count(1));
    }
  }
};

#define REGISTER_NMS_GPU_KERNEL(dtype)                                            \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kNonMaximumSuppressionConf, \
                                        DeviceType::kGPU, dtype, NmsGpuKernel<dtype>)

REGISTER_NMS_GPU_KERNEL(float);
REGISTER_NMS_GPU_KERNEL(double);

}  // namespace oneflow
