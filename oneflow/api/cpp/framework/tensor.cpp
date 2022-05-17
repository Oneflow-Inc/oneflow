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
#include "oneflow/api/cpp/framework/tensor.h"
#include "oneflow/api/cpp/framework/device.h"
#include "oneflow/api/cpp/framework/dtype.h"
#include "oneflow/api/cpp/framework/shape.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/api/common/ofblob.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/vm/virtual_machine.h"

namespace oneflow_api {

namespace of = oneflow;
namespace functional = of::one::functional;

Tensor::Tensor(const Shape& shape, const Device& device, const DType& dtype) {
  of::LazyMode::Guard lazy_mode_disabled_guard(/*is_enabled*/ false);
  tensor_ = functional::Empty(*shape.shape_,
                              of::DType::Get(static_cast<of::DataType>(dtype)).GetOrThrow(),
                              *device.device_, /*pin_memory=*/false)
                .GetPtrOrThrow();
}
Tensor::Tensor(const std::shared_ptr<oneflow::one::Tensor>& tensor) : tensor_(tensor) {}

Tensor::Tensor(const Tensor& tensor) : tensor_(tensor.tensor_) {}
Tensor::Tensor(Tensor&& tensor) noexcept : tensor_(std::move(tensor.tensor_)) {}

Tensor& Tensor::operator=(const Tensor& tensor) {
  if (&tensor == this) { return *this; }
  tensor_ = tensor.tensor_;
  return *this;
}
Tensor& Tensor::operator=(Tensor&& tensor) noexcept {
  if (&tensor == this) { return *this; }
  tensor_ = std::move(tensor.tensor_);
  return *this;
}

Shape Tensor::shape() const {
  const auto shape_ = tensor_->shape();
  return Shape(std::vector<int64_t>(shape_->dim_vec().begin(), shape_->dim_vec().end()));
}

Device Tensor::device() const {
  const auto device_ = tensor_->device().GetOrThrow();
  return Device(device_->type(), device_->device_id());
}

DType Tensor::dtype() const { return static_cast<DType>(tensor_->dtype()->data_type()); }

void Tensor::zeros_() {
  std::shared_ptr<of::one::MirroredTensor> local_tensor =
      tensor_->AsMirroredTensor().GetPtrOrThrow();
  of::PhysicalRun([&](of::InstructionsBuilder* builder) -> of::Maybe<void> {
    JUST(builder->AccessBlobByCallback(
        local_tensor,
        [](uint64_t of_blob_ptr) {
          auto* of_blob = reinterpret_cast<of::OfBlob*>(of_blob_ptr);
          of_blob->AsyncAutoMemset(0);
        },
        "mut"));
    return of::Maybe<void>::Ok();
  }).GetOrThrow();
}

Tensor Tensor::from_buffer(const void* buffer, const Shape& shape, const Device& device,
                           const DType& dtype) {
  Tensor tensor(shape, device, dtype);
  std::shared_ptr<of::one::MirroredTensor> local_tensor =
      tensor.tensor_->AsMirroredTensor().GetPtrOrThrow();
  of::PhysicalRun([&](of::InstructionsBuilder* builder) -> of::Maybe<void> {
    return builder->AccessBlobByCallback(
        local_tensor,
        [buffer, shape, dtype](uint64_t ofblob_ptr) {
          CHECK_JUST(of::BlobBufferCopyUtil<void>::From(ofblob_ptr, buffer,
                                                        shape.Count(0) * GetDTypeSize(dtype)));
        },
        "mut");
  }).GetOrThrow();
  return tensor;
}

template<typename T>
void Tensor::copy_to(T* buffer) const {
  std::shared_ptr<of::one::MirroredTensor> local_tensor =
      tensor_->AsMirroredTensor().GetPtrOrThrow();
  const auto shape = this->shape();

  const auto& Callback = [buffer, shape](uint64_t ofblob_ptr) {
    CHECK_JUST(of::BlobBufferCopyUtil<T>::To(ofblob_ptr, buffer, shape.Count(0)));
  };
  auto btb = std::make_shared<of::BlockingThenBusy>(1);
  CHECK_JUST(of::PhysicalRun([&](of::InstructionsBuilder* builder) -> of::Maybe<void> {
    return builder->SyncAccessBlobByCallback(local_tensor, btb, Callback, "const");
  }));
  TRY(btb->WaitUntilCntEqualZero(of::VirtualMachine::GetPredicatorNoMoreInstructionsFinished()))
      .GetOrThrow();
}

const std::shared_ptr<oneflow::one::Tensor>& Tensor::__internal_tensor() const { return tensor_; }

#define REGISTER_TENSOR_COPY_TO(cpp_dtype) \
  template void Tensor::copy_to<cpp_dtype>(cpp_dtype * buffer) const;

REGISTER_TENSOR_COPY_TO(float)
REGISTER_TENSOR_COPY_TO(double)
REGISTER_TENSOR_COPY_TO(bool)
REGISTER_TENSOR_COPY_TO(int8_t)
REGISTER_TENSOR_COPY_TO(int32_t)
REGISTER_TENSOR_COPY_TO(int64_t)

}  // namespace oneflow_api
