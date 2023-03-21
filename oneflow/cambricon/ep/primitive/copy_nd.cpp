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
    cnnlDataType_t cnnl_data_type = ConvertToCnnlDataType(data_type);
    CnnlTensorDescriptor input_desc, output_desc;
    if (num_dims == 0) {
      input_desc.set(num_dims, src_dims, cnnl_data_type);
      output_desc.set(num_dims, dst_dims, cnnl_data_type);
    } else {
      std::vector<int64_t> src_stride(num_dims, 1);
      std::vector<int64_t> dst_stride(num_dims, 1);

      int src_offset = src_pos[num_dims - 1];
      int dst_offset = dst_pos[num_dims - 1];
      for (int i = num_dims - 2; i >= 0; --i) {
        src_stride[i] = src_stride[i + 1] * src_dims[i + 1];
        dst_stride[i] = dst_stride[i + 1] * dst_dims[i + 1];
        src_offset += src_pos[i] * src_stride[i];
        dst_offset += dst_pos[i] * dst_stride[i];
      }

      input_desc.set(num_dims, extent, src_stride.data(), cnnl_data_type);
      output_desc.set(num_dims, extent, dst_stride.data(), cnnl_data_type);

      src = static_cast<const char*>(src) + src_offset * GetSizeOfDataType(data_type);
      dst = static_cast<char*>(dst) + dst_offset * GetSizeOfDataType(data_type);
    }

    OF_CNNL_CHECK(cnnlCopy(stream->As<ep::MluStream>()->cnnl_handle(), input_desc.desc(), src,
                           output_desc.desc(), dst));
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
