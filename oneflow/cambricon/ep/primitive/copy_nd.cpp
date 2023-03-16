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
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/ep/include/primitive/copy_nd.h"
#include "oneflow/core/ep/common/primitive/copy_nd.h"

namespace oneflow {
namespace ep {
namespace primitive {

namespace {

template<size_t num_dims, size_t movement_size, typename IndexType>
void LaunchKernel(Stream* stream, CopyNdKernelParams<num_dims, IndexType> params) {
  // Just to define LaunchKernel used in REGISTER_PRIMITIVE_FACTORY
  UNIMPLEMENTED();
}

class CopyNdImpl : public CopyNd {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyNdImpl);
  CopyNdImpl() = default;
  ~CopyNdImpl() override = default;

  void Launch(Stream* stream, DataType data_type, size_t num_dims, void* dst,
              const int64_t* dst_dims, const int64_t* dst_pos, const void* src,
              const int64_t* src_dims, const int64_t* src_pos,
              const int64_t* extent) const override {
    std::vector<int> begin(num_dims, 0);
    std::vector<int> end(num_dims, 0);
    std::vector<int> stride(num_dims, 1);
    for (int i = 0; i < num_dims; ++i) {
      begin[i] = static_cast<int>(src_pos[i]);
      end[i] = begin[i] + static_cast<int>(extent[i]);
    }
    cnnlDataType_t cnnl_data_type = ConvertToCnnlDataType(data_type);
    CnnlTensorDescriptor input_desc, output_desc;
    input_desc.set(num_dims, src_dims, cnnl_data_type);
    output_desc.set(num_dims, dst_dims, cnnl_data_type);
    OF_CNNL_CHECK(cnnlStridedSlice(stream->As<ep::MluStream>()->cnnl_handle(), input_desc.desc(),
                                   src, begin.data(), end.data(), stride.data(), output_desc.desc(),
                                   dst));
  }
};

class CopyNdFactoryImpl : public CopyNdFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyNdFactoryImpl);
  CopyNdFactoryImpl() = default;
  ~CopyNdFactoryImpl() override = default;

  std::unique_ptr<CopyNd> New(size_t max_num_dims) override {
    if (max_num_dims <= kMaxNumDims) {
      return std::unique_ptr<CopyNd>(new CopyNdImpl());
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kMLU, CopyNdFactory, CopyNdFactoryImpl);

}  // namespace

}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
