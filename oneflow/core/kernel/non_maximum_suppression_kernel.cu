#include "oneflow/core/kernel/non_maximum_suppression_kernel.h"
#include <cub/cub.cuh>

namespace oneflow {

namespace {

constexpr size_t kBlockSize = sizeof(int64_t) * 8;

template<typename T>
__host__ __device__ __forceinline__ T CeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

template<typename T>
__host__ __device__ __forceinline__ T IoU(T const* const a, T const* const b) {
  T interS = max(min(a[2], b[2]) - max(a[0], b[0]) + 1, 0.f)
             * max(min(a[3], b[3]) - max(a[1], b[1]) + 1, 0.f);
  T Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  T Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

template<typename T>
__global__ void CalcSuppressionBitmaskMatrix(size_t num_boxes, const float nms_iou_threshold,
                                             const T* boxes, int64_t* suppression_bmask_matrix) {
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

__global__ void ScanSuppression(const size_t num_boxes, const size_t num_blocks, size_t num_keep,
                                int64_t* suppression_bmask, int8_t* keep_mask) {
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

// class SuppressionChunkGetter final {
//  public:
//   SuppressionChunkGetter(int64_t* base, size_t count, size_t step) : base_block_(base),
//   count_(count), step_(step) {} CUB_RUNTIME_FUNCTION __forceinline__  operator()(size_t i) {
//     return base_block_ + (i % count_) * step_ + i / count_;
//   }
//  private:
//   int64_t* base_block_;
//   size_t count_;
//   size_t step_;
// };

// template<size_t N>
// struct SuppressionBitmaskChunk {
//   std::array<int64_t, N> bit_blocks_;

//   SuppressionBitmaskChunk() = delete;
//   ~SuppressionBitmaskChunk() = delete;
//   static const SuppressionBitmaskChunk<N>& Cast(const int64_t* ptr) {
//     return *reinterpret_cast<const SuppressionBitmaskChunk<N>*>(ptr);
//   }
//   static SuppressionBitmaskChunk<N>& Cast(int64_t* ptr) {
//     return *reinterpret_cast<SuppressionBitmaskChunk<N>*>(ptr);
//   }
// };

// template<size_t N>
// struct SuppressionScanBitwiseOrFunctor {
//   const int64_t* base_;

//   CUB_RUNTIME_FUNCTION __forceinline__ SuppressionBitmaskChunk<N>& operator()(
//       SuppressionBitmaskChunk<N>& a,
//       SuppressionBitmaskChunk<N>& b) {
//     size_t index = Index(reinterpret_cast<int64_t*>(&b));
//     size_t blockn = index / N;
//     size_t blocki = index % N;
//     if ((1ll << blocki) & a.at(index)) {
//       b = a;
//     } else {
// #pragma unroll
//       for (size_t i = blockn; i < N; ++i) {
//         b.at(i) |= a.at(i);
//       }
//     }
//     return b
//   }

//   CUB_RUNTIME_FUNCTION __forceinline__ size_t Index(int64_t* cur) const {
//     ptrdiff_t offset = cur - base_;
//     return offset / N;
//   }
// };

// void ScanSuppression(DeviceCtx* ctx, const int32_t num_boxes, int64_t* suppression_bmask,
//                             int64_t* scanned_suppression_bmask, void* temp_storage,
//                             size_t temp_storage_bytes) {
//   size_t num_blocks = static_cast<size_t>(CeilDiv(num_boxes, sizeof(int64_t)));
//   SuppressionScanBitwiseOrFunctor scan_op = {scanned_suppression_bmask};
//   cub::CountingInputIterator<ptrdiff_t> in_counting_iter(0);
//   cub::TransformInputIterator<int64_t*, BlockGetter, cub::CountingInputIterator<ptrdiff_t>>
//       in_iter(in_counting_iter, BlockGetter(suppression_bmask, num_boxes, num_blocks);
//   CudaCheck(cub::DeviceScan::ExclusiveScan(temp_storage, temp_storage_bytes, iou_matrix,
//   suppressions,
//                                            scan_op, 0ll, num_boxes, ctx->cuda_stream()));
// }

}  // namespace

template<typename T>
struct NonMaximumSuppressionUtil<DeviceType::kGPU, T> {
  static void Forward(DeviceCtx* ctx, const size_t num_boxes, const float nms_iou_threshold,
                      const size_t num_keep, const T* boxes, int64_t* suppression, int8_t* keep) {
    const size_t num_blocks = CeilDiv(num_boxes, kBlockSize);
    Memset<DeviceType::kGPU>(ctx, suppression, 0, num_boxes * num_blocks * sizeof(int64_t));
    Memset<DeviceType::kGPU>(ctx, keep, 0, num_boxes * sizeof(int8_t));
    dim3 blocks(num_blocks, num_blocks);
    dim3 threads(kBlockSize);
    CalcSuppressionBitmaskMatrix<<<blocks, threads, 0, ctx->cuda_stream()>>>(
        num_boxes, nms_iou_threshold, boxes, suppression);
    ScanSuppression<<<1, num_blocks, num_blocks, ctx->cuda_stream()>>>(num_boxes, num_blocks,
                                                                       num_keep, suppression, keep);
  }
};

#define MAKE_ENTRY(type_cpp, type_proto) \
  template struct NonMaximumSuppressionUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
