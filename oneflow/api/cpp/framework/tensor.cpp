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
#include "oneflow/core/common/thread_local_callback.h"
#include "oneflow/api/common/ofblob.h"
#include "oneflow/core/framework/dtype.h"

namespace oneflow_api {

namespace of = oneflow;
namespace functional = of::one::functional;

namespace {

// not all C++ compilers have default float so we define our own here
inline std::ios_base& defaultfloat(std::ios_base& __base) {
  __base.unsetf(std::ios_base::floatfield);
  return __base;
}

// saves/restores number formatting inside scope
struct FormatGuard {                                                                 // NOLINT
  FormatGuard(std::ostream& out) : out(out), saved(nullptr) { saved.copyfmt(out); }  // NOLINT
  ~FormatGuard() { out.copyfmt(saved); }

 private:
  std::ostream& out;
  std::ios saved;
};

std::tuple<double, int64_t> __printFormat(std::ostream& stream, const double* self_p, size_t size) {
  if (size == 0) { return std::make_tuple(1., 0); }
  bool intMode = true;
  for (size_t i = 0; i < size; ++i) {
    auto z = self_p[i];
    if (std::isfinite(z)) {
      if (z != std::ceil(z)) {
        intMode = false;
        break;
      }
    }
  }
  int64_t offset = 0;
  while (!std::isfinite(self_p[offset])) {
    offset = offset + 1;
    if (offset == size) { break; }
  }
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  double expMin;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  double expMax;
  if (offset == size) {
    expMin = 1;
    expMax = 1;
  } else {
    expMin = fabs(self_p[offset]);
    expMax = fabs(self_p[offset]);
    for (size_t i = offset; i < size; ++i) {
      double z = fabs(self_p[i]);
      if (std::isfinite(z)) {
        if (z < expMin) { expMin = z; }
        if (self_p[i] > expMax) { expMax = z; }
      }
    }
    if (expMin != 0) {
      expMin = std::floor(std::log10(expMin)) + 1;
    } else {
      expMin = 1;
    }
    if (expMax != 0) {
      expMax = std::floor(std::log10(expMax)) + 1;
    } else {
      expMax = 1;
    }
  }
  double scale = 1;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t sz;
  if (intMode) {
    if (expMax > 9) {
      sz = 11;
      stream << std::scientific << std::setprecision(4);
    } else {
      sz = expMax + 1;
      stream << defaultfloat;
    }
  } else {
    if (expMax - expMin > 4) {
      sz = 11;
      if (std::fabs(expMax) > 99 || std::fabs(expMin) > 99) { sz = sz + 1; }
      stream << std::scientific << std::setprecision(4);
    } else {
      if (expMax > 5 || expMax < 0) {
        sz = 7;
        scale = std::pow(10, expMax - 1);
        stream << std::fixed << std::setprecision(4);
      } else {
        if (expMax == 0) {
          sz = 7;
        } else {
          sz = expMax + 6;
        }
        stream << std::fixed << std::setprecision(4);
      }
    }
  }
  return std::make_tuple(scale, sz);
}

void __printIndent(std::ostream& stream, int64_t indent) {
  for (size_t i = 0; i < indent; ++i) {
    (void)i;  // Suppress unused variable warning
    stream << " ";
  }
}

void printScale(std::ostream& stream, double scale) {
  FormatGuard guard(stream);
  stream << defaultfloat << scale << " *" << std::endl;
}

void __printMatrix(std::ostream& stream, const double* self, const Shape& shape, int64_t linesize,
                   int64_t indent) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  double scale;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t sz;
  std::tie(scale, sz) = __printFormat(stream, self, shape.Count(0));
  std::tie(scale, sz) = __printFormat(stream, self, shape.Count(0));

  __printIndent(stream, indent);
  int64_t nColumnPerLine = (linesize - indent) / (sz + 1);
  int64_t firstColumn = 0;
  int64_t lastColumn = -1;
  while (firstColumn < shape.At(1)) {
    if (firstColumn + nColumnPerLine <= shape.At(1)) {
      lastColumn = firstColumn + nColumnPerLine - 1;
    } else {
      lastColumn = shape.At(1) - 1;
    }
    if (nColumnPerLine < shape.At(1)) {
      if (firstColumn != 0) { stream << std::endl; }
      stream << "Columns " << firstColumn + 1 << " to " << lastColumn + 1 << "\n";
      __printIndent(stream, indent);
    }
    if (scale != 1) {
      printScale(stream, scale);
      __printIndent(stream, indent);
    }
    for (size_t l = 0; l < shape.At(0); ++l) {
      const double* row_ptr = self + l * (shape.Count(1));
      for (int64_t c = firstColumn; c < lastColumn + 1; c++) {
        stream << std::setw(sz) << row_ptr[c] / scale;
        if (c == lastColumn) {
          stream << std::endl;
          if (l != shape.At(0) - 1) {
            if (scale != 1) {
              __printIndent(stream, indent);
              stream << " ";
            } else {
              __printIndent(stream, indent);
            }
          }
        } else {
          stream << " ";
        }
      }
    }
    firstColumn = lastColumn + 1;
  }
}

void __printTensor(std::ostream& stream, const double* self, const Shape& shape, int64_t linesize) {
  std::vector<int64_t> counter(shape.NumAxes() - 2);
  bool start = true;
  bool finished = false;
  counter[0] = -1;
  for (size_t i = 1; i < counter.size(); ++i) { counter[i] = 0; }
  const auto ori_dim_vec = shape.DimVec();
  while (true) {
    for (int64_t i = 0; shape.NumAxes() - 2; i++) {
      counter[i] = counter[i] + 1;
      if (counter[i] >= shape.At(i)) {
        if (i == shape.NumAxes() - 3) {
          finished = true;
          break;
        }
        counter[i] = 0;
      } else {
        break;
      }
    }
    if (finished) { break; }
    if (start) {
      start = false;
    } else {
      stream << std::endl;
    }
    stream << "(";
    int64_t offset = 0;
    int64_t t = 0;
    for (; t < shape.NumAxes() - 2; t++) {
      offset += counter[t] * shape.Count(t + 1);
      stream << counter[t] + 1 << ",";
    }
    stream << ".,.) = " << std::endl;

    __printMatrix(stream, self + offset,
                  Shape(std::vector<int64_t>(ori_dim_vec.begin() + t, ori_dim_vec.end())), linesize,
                  1);
  }
}

std::ostream& print(std::ostream& stream, const double* data, const Shape& shape, int64_t n_dims,
                    int64_t n_elems, int64_t linesize) {
  if (n_dims == 0) {
    stream << defaultfloat << data[0] << std::endl;
    stream << "[ "
           << "{}";
  } else if (n_dims == 1) {
    if (n_elems > 0) {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      double scale;
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      int64_t sz;
      std::tie(scale, sz) = __printFormat(stream, data, n_elems);
      if (scale != 1) { printScale(stream, scale); }
      for (auto i = 0; i < n_dims; ++i) { stream << std::setw(sz) << data[i] / scale << std::endl; }
    }
  } else if (n_dims == 2) {
    if (n_elems > 0) { __printMatrix(stream, data, shape, linesize, 0); }
  } else {
    if (n_elems > 0) { __printTensor(stream, data, shape, linesize); }
  }
  return stream;
}

template<typename T>
std::ostream& print(std::ostream& stream, const Tensor& tensor, int64_t linesize) {
  FormatGuard guard(stream);

  const Shape shape = tensor.shape();
  const auto n_elems = tensor.shape().Count(0);

  std::vector<double> data;
  {
    std::vector<T> temp_data(n_elems, 0);
    tensor.copy_to(temp_data.data());
    std::vector<double>(n_elems, 0).swap(data);
    for (size_t i = 0; i < data.size(); ++i) { data[i] = static_cast<double>(temp_data[i]); }
  }

  print(stream, data.data(), shape, shape.NumAxes(), n_elems, linesize);
  return stream;
}

template<>
std::ostream& print<double>(std::ostream& stream, const Tensor& tensor, int64_t linesize) {
  FormatGuard guard(stream);

  const Shape shape = tensor.shape();
  const auto n_elems = tensor.shape().Count(0);

  std::vector<double> data(n_elems, 0);
  tensor.copy_to(data.data());

  print(stream, data.data(), shape, shape.NumAxes(), n_elems, linesize);
  return stream;
}

}  // namespace

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
  const int64_t linesize = 80;
  std::map<DType, std::function<std::ostream&()>> f{
      {DType::kInt8, [&]() { return std::ref(print<int8_t>(os, tensor, linesize)); }},
      {DType::kInt32, [&]() { return std::ref(print<int32_t>(os, tensor, linesize)); }},
      {DType::kInt64, [&]() { return std::ref(print<int64_t>(os, tensor, linesize)); }},
      {DType::kFloat, [&]() { return std::ref(print<float>(os, tensor, linesize)); }},
      {DType::kDouble, [&]() { return std::ref(print<double>(os, tensor, linesize)); }}};
  f[tensor.dtype()]();
  os << "["
     << "Shape: " << tensor.shape() << ", Device: " << tensor.device()
     << ", DataType: " << tensor.dtype() << "]";
  return os;
}

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

  const auto& Callback =
      std::make_shared<std::function<void(uint64_t)>>([buffer, shape](uint64_t ofblob_ptr) {
        CHECK_JUST(of::BlobBufferCopyUtil<T>::To(ofblob_ptr, buffer, shape.Count(0)));
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

#define REGISTER_TENSOR_COPY_TO(cpp_dtype) \
  template void Tensor::copy_to<cpp_dtype>(cpp_dtype * buffer) const;

REGISTER_TENSOR_COPY_TO(float)
REGISTER_TENSOR_COPY_TO(double)
REGISTER_TENSOR_COPY_TO(int8_t)
REGISTER_TENSOR_COPY_TO(int32_t)
REGISTER_TENSOR_COPY_TO(int64_t)

}  // namespace oneflow_api
