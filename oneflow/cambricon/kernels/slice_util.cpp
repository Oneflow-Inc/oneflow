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
#include "oneflow/cambricon/kernels/slice_util.h"

#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"

namespace oneflow {
namespace mlu {

bool IsContiguous(size_t num_dims, const int64_t* dims, const int64_t* strides) {
  for (int i = num_dims - 1; i >= 0; i--) {
    if ((i == num_dims - 1 && strides[i] != 1)
        || (i != num_dims - 1 && strides[i] != dims[i + 1] * strides[i + 1])) {
      return false;
    }
  }
  return true;
}

void SliceKernelUtil::Forward(ep::Stream* stream, const SliceParams& params, DataType data_type,
                              const void* entire, void* sliced) {
  std::vector<int> begin(params.ndim, 0);
  std::vector<int> end(params.ndim, 0);
  std::vector<int> stride(params.ndim, 1);
  for (int i = 0; i < params.ndim; ++i) {
    begin[i] = params.start[i];
    stride[i] = params.step[i];
    end[i] = (params.size[i] - 1) * stride[i] + begin[i] + 1;
  }
  cnnlDataType_t cnnl_data_type = ConvertToCnnlDataType(data_type);
  CnnlTensorDescriptor input_desc, output_desc;
  input_desc.set(params.ndim, params.dims, params.stride, cnnl_data_type);
  output_desc.set(params.ndim, params.size, cnnl_data_type);
  OF_CNNL_CHECK(cnnlStridedSlice(stream->As<ep::MluStream>()->cnnl_handle(), input_desc.desc(),
                                 entire, begin.data(), end.data(), stride.data(),
                                 output_desc.desc(), sliced));
}

void SliceKernelUtil::Forward(ep::Stream* stream, const SliceParams& entire_params,
                              const SliceParams& sliced_params, DataType data_type,
                              const void* entire, void* sliced) {
  std::vector<int> begin(entire_params.ndim, 0);
  std::vector<int> end(entire_params.ndim, 0);
  std::vector<int> stride(entire_params.ndim, 1);
  for (int i = 0; i < entire_params.ndim; ++i) {
    begin[i] = entire_params.start[i];
    stride[i] = entire_params.step[i];
    end[i] = (entire_params.size[i] - 1) * stride[i] + begin[i] + 1;
  }
  cnnlDataType_t cnnl_data_type = ConvertToCnnlDataType(data_type);
  CnnlTensorDescriptor input_desc, output_desc;

  int64_t element_size = GetSizeOfDataType(data_type);
  int64_t sliced_start = 0;
  std::vector<int64_t> sliced_stride(sliced_params.ndim);
  for (int i = 0; i < sliced_params.ndim; ++i) {
    sliced_stride[i] = sliced_params.step[i] * sliced_params.stride[i];
    sliced_start += sliced_params.start[i] * sliced_params.stride[i];
  }
  sliced_start *= element_size;
  output_desc.set(sliced_params.ndim, sliced_params.size, sliced_stride.data(), cnnl_data_type);

  auto input_has_0_stride = [&]() {
    for (int i = 0; i < entire_params.ndim; ++i) {
      if (entire_params.stride[i] != 0) { return false; }
    }
    return true;
  }();
  if (input_has_0_stride) {
    OF_CNNL_CHECK(cnnlFill_v3(stream->As<ep::MluStream>()->cnnl_handle(), CNNL_POINTER_MODE_DEVICE,
                              entire, output_desc.desc(),
                              reinterpret_cast<char*>(sliced) + sliced_start));
    return;
  }

  CnnlWorkspace temp_entire(stream->As<ep::MluStream>(), 0);
  const void* contiguous_entire = entire;
  input_desc.set(entire_params.ndim, entire_params.dims, entire_params.stride, cnnl_data_type);
  // cnnlStridedSlice does not support non-contiguous input.
  if (!IsContiguous(entire_params.ndim, entire_params.dims, entire_params.stride)) {
    temp_entire.resize(entire_params.elem_cnt() * element_size);
    CnnlTensorDescriptor temp_entire_desc;
    temp_entire_desc.set(entire_params.ndim, entire_params.size, cnnl_data_type);

    OF_CNNL_CHECK(cnnlCopy(stream->As<ep::MluStream>()->cnnl_handle(), input_desc.desc(), entire,
                           temp_entire_desc.desc(), temp_entire.dptr()));
    contiguous_entire = temp_entire.dptr();
    input_desc = std::move(temp_entire_desc);
  }

  // cnnlStridedSlice does not support non-contiguous output.
  if (IsContiguous(sliced_params.ndim, sliced_params.size, sliced_stride.data())) {
    OF_CNNL_CHECK(cnnlStridedSlice(stream->As<ep::MluStream>()->cnnl_handle(), input_desc.desc(),
                                   contiguous_entire, begin.data(), end.data(), stride.data(),
                                   output_desc.desc(),
                                   reinterpret_cast<char*>(sliced) + sliced_start));
  } else {
    CnnlWorkspace temp(stream->As<ep::MluStream>(), sliced_params.elem_cnt() * element_size);
    CnnlTensorDescriptor temp_desc;
    temp_desc.set(sliced_params.ndim, sliced_params.size, cnnl_data_type);
    OF_CNNL_CHECK(cnnlStridedSlice(stream->As<ep::MluStream>()->cnnl_handle(), input_desc.desc(),
                                   contiguous_entire, begin.data(), end.data(), stride.data(),
                                   temp_desc.desc(), temp.dptr()));
    OF_CNNL_CHECK(cnnlCopy(stream->As<ep::MluStream>()->cnnl_handle(), temp_desc.desc(),
                           temp.dptr(), output_desc.desc(),
                           reinterpret_cast<char*>(sliced) + sliced_start));
  }
}

}  // namespace mlu
}  // namespace oneflow
