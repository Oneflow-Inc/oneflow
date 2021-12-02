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
#include <cstddef>
#include <memory>
#include <utility>
#include "oneflow/api/cpp/framework/device.h"
#include "oneflow/api/cpp/framework/dtype.h"
#include "oneflow/api/cpp/framework/shape.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/common/thread_local_callback.h"
#include "oneflow/api/common/ofblob.h"
#include "oneflow/core/framework/dtype.h"

namespace oneflow_api {

namespace of = oneflow;
namespace functional = of::one::functional;

Tensor::Tensor(const Shape& shape, const Device& device, const DType& dtype) {
  of::LazyMode::Guard lazy_mode_disabled_guard(/*is_enabled*/ false);
  tensor_ = functional::Empty(*shape.shape_,
                              of::DType::Get(static_cast<of::DataType>(dtype)).GetOrThrow(),
                              *device.device_)
                .GetPtrOrThrow();
}
Tensor::Tensor(const std::shared_ptr<oneflow::one::Tensor>& tensor) : tensor_(tensor) {}

Tensor::Tensor(const Tensor& tensor) : tensor_(tensor.tensor_) {}
Tensor::Tensor(Tensor&& tensor) noexcept : tensor_(std::move(tensor.tensor_)) {}

Tensor::~Tensor() {}

Tensor& Tensor::operator=(const Tensor& tensor) {
  tensor_.reset();
  tensor_ = tensor.tensor_;
  return *this;
}
Tensor& Tensor::operator=(Tensor&& tensor) noexcept {
  tensor_.reset();
  tensor_ = std::move(tensor.tensor_);
  return *this;
}

const Shape Tensor::shape() const {
  const auto shape_ = tensor_->shape();
  return Shape(std::vector<int64_t>(shape_->dim_vec().begin(), shape_->dim_vec().end()));
}

const Device Tensor::device() const {
  const auto device_ = tensor_->device().GetOrThrow();
  return Device(device_->type(), device_->device_id());
}

const DType Tensor::dtype() const { return static_cast<DType>(tensor_->dtype()->data_type()); }

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

Tensor Tensor::from_blob(const void* blob, const Shape& shape, const Device& device,
                         const DType& dtype) {
  Tensor tensor(shape, device, dtype);
  std::shared_ptr<of::one::MirroredTensor> local_tensor =
      tensor.tensor_->AsMirroredTensor().GetPtrOrThrow();
  of::PhysicalRun([&](of::InstructionsBuilder* builder) -> of::Maybe<void> {
    return builder->AccessBlobByCallback(
        local_tensor,
        [blob, shape, dtype](uint64_t ofblob_ptr) {
          CHECK_JUST(of::BlobBufferCopyUtil<char>::From(ofblob_ptr, static_cast<const char*>(blob),
                                                        shape.Count(0) * GetDTypeSize(dtype)));
        },
        "mut");
  }).GetOrThrow();
  return tensor;
}

template<typename T>
void Tensor::copy_to(T* blob) {
  std::shared_ptr<of::one::MirroredTensor> local_tensor =
      tensor_->AsMirroredTensor().GetPtrOrThrow();
  const auto shape = this->shape();

  const auto& Callback =
      std::make_shared<std::function<void(uint64_t)>>([blob, shape](uint64_t ofblob_ptr) {
        CHECK_JUST(of::BlobBufferCopyUtil<T>::To(ofblob_ptr, blob, shape.Count(0)));
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

const std::shared_ptr<oneflow::one::Tensor>& Tensor::__internal_tensor() const { return tensor_; }

#define REGISTER_TO_BLOB(cpp_dtype) template void Tensor::copy_to<cpp_dtype>(cpp_dtype * blob);

REGISTER_TO_BLOB(float)
REGISTER_TO_BLOB(double)
REGISTER_TO_BLOB(int8_t)
REGISTER_TO_BLOB(int32_t)
REGISTER_TO_BLOB(int64_t)

}  // namespace oneflow_api
