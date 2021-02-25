#ifndef ONEFLOW_CORE_DEVICE_CUDA_STREAM_INDEX_H_
#define ONEFLOW_CORE_DEVICE_CUDA_STREAM_INDEX_H_

#include "oneflow/core/device/stream_index.h"

namespace oneflow {

class CudaStreamIndexGenerator final : public StreamIndexGenerator {
 public:
  stream_index_t GenerateComputeStreamIndex() override { return kCompute; }
  stream_index_t GenerateH2DStreamIndex() override { return kH2D; }
  stream_index_t GenerateD2HStreamIndex() override { return kD2H; }
  stream_index_t GenerateMixStreamIndex() { return kMix; }
  stream_index_t GenerateNcclStreamIndex() { return kNccl; }
  stream_index_t GenerateDecodeH2DStreamIndex() { return kDecodeH2D; }

  bool IsComputeStreamIndex(stream_index_t index) const override { return index == kCompute; }
  bool IsH2DStreamIndex(stream_index_t index) const override { return index == kH2D; }
  bool IsD2HStreamIndex(stream_index_t index) const override { return index == kD2H; }
  bool IsMixStreamIndex(stream_index_t index) const { return index == kMix; }
  bool IsNcclStreamIndex(stream_index_t index) const { return index == kNccl; }
  bool IsDecodeH2DStreamIndex(stream_index_t index) const { return index == kDecodeH2D; }

 private:
  static const stream_index_t kCompute = 0;
  static const stream_index_t kH2D = 1;
  static const stream_index_t kD2H = 2;
  static const stream_index_t kMix = 3;
  static const stream_index_t kNccl = 4;
  static const stream_index_t kDecodeH2D = 5;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_CUDA_STREAM_INDEX_H_
