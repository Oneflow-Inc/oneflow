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
#include "oneflow/core/ep/include/primitive/permute.h"
#include "oneflow/core/ep/common/primitive/permute_impl.h"

namespace oneflow {

namespace ep {
namespace primitive {

namespace permute {

namespace internal {

namespace {

template<size_t num_dims, size_t movement_size, typename IndexType>
void PermuteKernel(PermuteKernelParams<num_dims, IndexType> params) {
  using T = typename std::aligned_storage<movement_size, movement_size>::type;
  const T* src = reinterpret_cast<const T*>(params.src);
  T* dst = reinterpret_cast<T*>(params.dst);
  for (IndexType i = 0; i < params.count; ++i) {
    IndexType src_index[num_dims];
    IndexType dst_index[num_dims];
    params.dst_index_helper.OffsetToNdIndex(i, dst_index);
    for (size_t dim = 0; dim < num_dims; ++dim) {
      src_index[params.permutation[dim]] = dst_index[dim];
    }
    IndexType src_offset = params.src_index_helper.NdIndexToOffset(src_index);
    dst[i] = src[src_offset];
  }
}

template<size_t num_dims, size_t movement_size, typename IndexType>
void LaunchKernel(Stream* stream, const int64_t* src_dims, const void* src, const int* permutation,
                  void* dst, size_t count) {
  PermuteKernelParams<num_dims, IndexType> params =
      MakePermuteParams<num_dims, IndexType>(src_dims, src, permutation, dst, count);
  PermuteKernel<num_dims, movement_size, IndexType>(params);
}
class PermuteImpl : public Permute {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PermuteImpl);
  PermuteImpl() = default;
  ~PermuteImpl() override = default;

  using Permute::Launch;
  void Launch(Stream* stream, DataType data_type, size_t num_dims, const int64_t* src_dims,
              const void* src, const int* permutation, void* dst) override {
    SimplifyThenLaunch(stream, data_type, num_dims, src_dims, src, permutation, dst);
  }
};

class PermuteFactoryImpl : public PermuteFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PermuteFactoryImpl);
  PermuteFactoryImpl() = default;
  ~PermuteFactoryImpl() override = default;

  std::unique_ptr<Permute> New(size_t max_num_dims) override {
    if (max_num_dims <= kMaxNumDims) {
      return std::unique_ptr<Permute>(new PermuteImpl());
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, PermuteFactory, PermuteFactoryImpl);

}  // namespace

}  // namespace internal

}  // namespace permute

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
