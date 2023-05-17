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
#include "oneflow/core/ep/include/primitive/batch_matmul.h"
#include "oneflow/core/ep/include/primitive/broadcast_matmul.h"

namespace oneflow {

namespace ep {
namespace primitive {

namespace {

class BatchMatmulImpl : public BatchMatmul {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BatchMatmulImpl);
  BatchMatmulImpl(BlasTransposeType transpose_a, BlasTransposeType transpose_b,
                  std::unique_ptr<BroadcastMatmul>&& broadcast_matmul)
      : transpose_a_(transpose_a),
        transpose_b_(transpose_b),
        broadcast_matmul_(std::move(broadcast_matmul)) {}
  ~BatchMatmulImpl() override = default;

  void Launch(Stream* stream, size_t batch_size, size_t m, size_t n, size_t k, Scalar alpha,
              const void* a, const void* b, Scalar beta, void* c) override {
    int64_t a_dims[3];
    int64_t b_dims[3];
    int64_t c_dims[3];
    a_dims[0] = batch_size;
    b_dims[0] = batch_size;
    c_dims[0] = batch_size;
    if (transpose_a_ == BlasTransposeType::N) {
      a_dims[1] = m;
      a_dims[2] = k;
    } else if (transpose_a_ == BlasTransposeType::T) {
      a_dims[1] = k;
      a_dims[2] = m;
    } else {
      UNIMPLEMENTED();
    }
    if (transpose_b_ == BlasTransposeType::N) {
      b_dims[1] = k;
      b_dims[2] = n;
    } else if (transpose_b_ == BlasTransposeType::T) {
      b_dims[1] = n;
      b_dims[2] = k;
    } else {
      UNIMPLEMENTED();
    }
    c_dims[1] = m;
    c_dims[2] = n;
    broadcast_matmul_->Launch(stream, alpha, 3, a_dims, a, 3, b_dims, b, beta, 3, c_dims, c);
  }

 private:
  BlasTransposeType transpose_a_;
  BlasTransposeType transpose_b_;
  std::unique_ptr<BroadcastMatmul> broadcast_matmul_;
};

template<DeviceType device_type>
class BatchMatmulFactoryImpl : public BatchMatmulFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BatchMatmulFactoryImpl);
  BatchMatmulFactoryImpl() = default;
  ~BatchMatmulFactoryImpl() override = default;

  std::unique_ptr<BatchMatmul> New(DataType data_type, BlasTransposeType transpose_a,
                                   BlasTransposeType transpose_b) override {
    auto broadcast_matmul =
        NewPrimitive<BroadcastMatmulFactory>(device_type, data_type, transpose_a, transpose_b, 3);
    if (!broadcast_matmul) { return nullptr; }
    return std::make_unique<BatchMatmulImpl>(transpose_a, transpose_b, std::move(broadcast_matmul));
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, BatchMatmulFactory,
                           BatchMatmulFactoryImpl<DeviceType::kCPU>);

#ifdef WITH_CUDA
REGISTER_PRIMITIVE_FACTORY(DeviceType::kCUDA, BatchMatmulFactory,
                           BatchMatmulFactoryImpl<DeviceType::kCUDA>);
#endif  // WITH_CUDA

}  // namespace

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
