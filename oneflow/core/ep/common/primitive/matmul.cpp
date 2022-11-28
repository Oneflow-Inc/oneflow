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
#include "oneflow/core/ep/include/primitive/matmul.h"
#include "oneflow/core/ep/include/primitive/batch_matmul.h"

namespace oneflow {

namespace ep {
namespace primitive {

namespace {

class MatmulImpl : public Matmul {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MatmulImpl);
  explicit MatmulImpl(std::unique_ptr<BatchMatmul>&& batch_matmul)
      : batch_matmul_(std::move(batch_matmul)) {}
  ~MatmulImpl() override = default;

  void Launch(Stream* stream, size_t m, size_t n, size_t k, Scalar alpha, const void* a,
              const void* b, Scalar beta, void* c) override {
    batch_matmul_->Launch(stream, 1, m, n, k, alpha, a, b, beta, c);
  }

 private:
  std::unique_ptr<BatchMatmul> batch_matmul_;
};

template<DeviceType device_type>
class MatmulFactoryImpl : public MatmulFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MatmulFactoryImpl);
  MatmulFactoryImpl() = default;
  ~MatmulFactoryImpl() override = default;

  std::unique_ptr<Matmul> New(DataType data_type, BlasTransposeType transpose_a,
                              BlasTransposeType transpose_b) override {
    auto batch_matmul =
        NewPrimitive<BatchMatmulFactory>(device_type, data_type, transpose_a, transpose_b);
    if (!batch_matmul) { return nullptr; }
    return std::make_unique<MatmulImpl>(std::move(batch_matmul));
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, MatmulFactory, MatmulFactoryImpl<DeviceType::kCPU>);

#ifdef WITH_CUDA
REGISTER_PRIMITIVE_FACTORY(DeviceType::kCUDA, MatmulFactory, MatmulFactoryImpl<DeviceType::kCUDA>);
#endif  // WITH_CUDA

}  // namespace

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
