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
#include "tensor.h"
#include "device.h"
#include "shape.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/common/thread_local_callback.h"

namespace oneflow_api {

namespace of = oneflow;
namespace functional = of::one::functional;

namespace {
struct OfBlobCopyBuffer {
  template<typename T>
  static of::Maybe<void> From(uint64_t of_blob_ptr, const T* buf_ptr, size_t size) {
    auto* of_blob = reinterpret_cast<of::OfBlob*>(of_blob_ptr);
    of_blob->AutoMemCopyFrom<T>(buf_ptr, size);
    return of::Maybe<void>::Ok();
  }

  template<typename T>
  static of::Maybe<void> To(uint64_t of_blob_ptr, T* buf_ptr, size_t size) {
    auto* of_blob = reinterpret_cast<of::OfBlob*>(of_blob_ptr);
    of_blob->AutoMemCopyTo<T>(buf_ptr, size);
    return of::Maybe<void>::Ok();
  }
};
}  // namespace

Tensor::Tensor() : Tensor(Device("cpu")) {}

Tensor::Tensor(const Device& device) : Tensor(Shape(), device) {}

Tensor::Tensor(const Shape& shape, const Device& device) {
  of::LazyMode::Guard lazy_mode_disabled_guard(/*is_enabled*/ false);
  tensor_ = functional::Empty(*shape.shape_, of::DType::Double(), *device.device_).GetPtrOrThrow();
}

const Shape Tensor::shape() const {
  const auto shape_ = tensor_->shape();
  return Shape(DimVector(shape_->dim_vec().begin(), shape_->dim_vec().end()));
}

const Device Tensor::device() const {
  const auto device_ = tensor_->device().GetOrThrow();
  return Device(device_->type(), device_->device_id());
}

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

template<typename T>
Tensor Tensor::from_blob(const T* blob, const Shape& shape, const Device& device) {
  Tensor tensor(shape, device);
  std::shared_ptr<of::one::MirroredTensor> local_tensor =
      tensor.tensor_->AsMirroredTensor().GetPtrOrThrow();
  of::PhysicalRun([&](of::InstructionsBuilder* builder) -> of::Maybe<void> {
    return builder->AccessBlobByCallback(
        local_tensor,
        [blob, shape](uint64_t ofblob_ptr) {
          CHECK_JUST(OfBlobCopyBuffer::template From<T>(ofblob_ptr, blob, shape.Count(0)));
        },
        "mut");
  }).GetOrThrow();
  return tensor;
}

template<typename T>
void Tensor::to_blob(const Tensor& tensor, T* blob) {
  std::shared_ptr<of::one::MirroredTensor> local_tensor =
      tensor.tensor_->AsMirroredTensor().GetPtrOrThrow();
  const auto shape = tensor.shape();

  const auto& Callback =
      std::make_shared<std::function<void(uint64_t)>>([blob, shape](uint64_t ofblob_ptr) {
        CHECK_JUST(OfBlobCopyBuffer::template To<T>(ofblob_ptr, blob, shape.Count(0)));
      });

  bool is_printed = false;
  of::SpinCounter::SpinWait(
      1,
      [&](const std::shared_ptr<of::SpinCounter>& sc) -> of::Maybe<void> {
        return of::PhysicalRun([&](of::InstructionsBuilder* builder) -> of::Maybe<void> {
          return builder->SyncAccessBlobByCallback(local_tensor, sc, Callback, "const");
        });
      },
      [&is_printed]() {
        if (!is_printed) {
          of::blocking::StackInfoCallback();
          is_printed = true;
        }
      })
      .GetOrThrow();
}

template Tensor Tensor::from_blob<double>(const double* blob, const Shape& shape,
                                          const Device& device);

template void Tensor::to_blob<double>(const Tensor& tensor, double* blob);

}  // namespace oneflow_api
