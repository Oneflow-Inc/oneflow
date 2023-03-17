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
#include "oneflow/core/ep/common/primitive/permute.h"
#include "oneflow/core/ep/common/primitive/permute_impl.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_op_descriptor.h"

namespace oneflow {
namespace ep {
namespace primitive {

namespace {

class PermuteImpl : public Permute {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PermuteImpl);
  PermuteImpl() = default;
  ~PermuteImpl() override = default;
  using Permute::Launch;
  void Launch(Stream* stream, DataType data_type, size_t num_dims, const int64_t* src_dims,
              const void* src, const int* permutation, void* dst) override {
    CnnlTransposeDescriptor tran_desc;

    tran_desc.set(num_dims, permutation);
    int64_t dst_dims[num_dims];
    for (size_t i = 0; i < num_dims; ++i) { dst_dims[i] = src_dims[permutation[i]]; }

    cnnlDataType_t cnnl_data_type = ConvertToCnnlDataType(data_type);
    CnnlTensorDescriptor input_desc, output_desc;
    input_desc.set(num_dims, src_dims, cnnl_data_type);
    output_desc.set(num_dims, dst_dims, cnnl_data_type);

    size_t workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetTransposeWorkspaceSize(stream->As<ep::MluStream>()->cnnl_handle(),
                                                input_desc.desc(), tran_desc.desc(),
                                                &workspace_size));
    CnnlWorkspace cnnl_workspace(stream->As<ep::MluStream>(), workspace_size);
    void* transpose_workspace = cnnl_workspace.dptr();

    OF_CNNL_CHECK(cnnlTranspose_v2(stream->As<ep::MluStream>()->cnnl_handle(), tran_desc.desc(),
                                   input_desc.desc(), src, output_desc.desc(), dst,
                                   transpose_workspace, workspace_size));
  }
};

class PermuteFactoryImpl : public PermuteFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PermuteFactoryImpl);
  PermuteFactoryImpl() = default;
  ~PermuteFactoryImpl() override = default;

  std::unique_ptr<Permute> New(size_t max_num_dims) override {
    return std::unique_ptr<Permute>(new PermuteImpl());
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kMLU, PermuteFactory, PermuteFactoryImpl);

}  // namespace

}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
