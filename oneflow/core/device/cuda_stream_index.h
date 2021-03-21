/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
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
