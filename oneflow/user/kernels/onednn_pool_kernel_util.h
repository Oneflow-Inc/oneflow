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
#ifndef ONEFLOW_USER_KERNELS_ONEDNN_POOL_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_ONEDNN_POOL_KERNEL_UTIL_H_
#ifdef WITH_ONEDNN

#include "oneflow/core/ep/cpu/cpu_stream.h"
#include "oneflow/core/ep/cpu/cpu_device.h"
#include "oneflow/user/kernels/max_pool_kernel_util.h"

namespace oneflow {

template<typename T>
dnnl::memory::data_type CppTypeToOneDnnDtype();

template<>
dnnl::memory::data_type CppTypeToOneDnnDtype<int32_t>() {
  return dnnl::memory::data_type::s32;
}

template<>
dnnl::memory::data_type CppTypeToOneDnnDtype<float>() {
  return dnnl::memory::data_type::f32;
}

template<typename T>
struct OneDnnPoolKernelUtil {
  static void OneDnnPool1dForwardCompute(
      ep::Stream* stream, const dnnl::memory::dims src_dims, const dnnl::memory::dims dst_dims,
      const dnnl::memory::dims kernel_dims, const dnnl::memory::dims strides_dims,
      const dnnl::memory::dims padding_dims_l, const dnnl::memory::dims padding_dims_r,
      const dnnl::memory::dims dilation, const void* src, void* dest, int64_t* indice_ptr) {
    auto data_type = CppTypeToOneDnnDtype<T>();
    ep::CpuStream* cpu_stream = stream->As<ep::CpuStream>();
    size_t num_threads = cpu_stream->device()->GetNumThreads();
    ep::CpuNumThreadsGuard guard(num_threads);
    dnnl::engine* onednn_engine = cpu_stream->onednn_engine();
    dnnl::stream* onednn_stream = cpu_stream->onednn_stream();

    // const dnnl::memory::dim n = 1, input_channel = 1,
    //         input_height = batch_size, input_width = width,
    //         kernel_height = 1, kernel_width = params_3d.pool_size_3d()[2],
    //         padding_height_left = 0,  padding_height_right = 0,
    //         padding_width_left = params_3d.padding()[2],
    //         padding_width_right = params_3d.padding()[2],
    //         stride_height = 1, stride_width = params_3d.stride_3d()[2],
    //         dilation_height = 1,dilation_width = 1;

    // const dnnl::memory::dim output_height = (input_height - ((kernel_height - 1) *
    // dilation_height+ kernel_height) + padding_height_left + padding_height_right) / stride_height
    // + 1; const dnnl::memory::dim output_width = (input_width - ((kernel_width - 1) *
    // dilation_width + kernel_width) + padding_width_left + padding_width_right) / stride_width +
    // 1;

    // dnnl::memory::dims kernel_dims = {kernel_height, kernel_width};
    // dnnl::memory::dims src_dims = {n, input_channel, input_height, input_width};
    // dnnl::memory::dims dst_dims = {n, input_channel, output_height, output_width};
    // dnnl::memory::dims strides_dims = {stride_height, stride_width};
    // dnnl::memory::dims padding_dims_l = {padding_height_left, padding_width_left};
    // dnnl::memory::dims padding_dims_r = {padding_height_right, padding_width_right};
    // dnnl::memory::dims dilation = {dilation_height, dilation_width};

    auto src_md = dnnl::memory::desc(src_dims, data_type, dnnl::memory::format_tag::nchw);
    auto dst_md = dnnl::memory::desc(dst_dims, data_type, dnnl::memory::format_tag::nchw);
    auto src_mem = dnnl::memory(src_md, *onednn_engine, (void*)src);
    auto dst_mem = dnnl::memory(dst_md, *onednn_engine, (void*)dest);

    auto pooling_d = dnnl::pooling_v2_forward::desc(
        dnnl::prop_kind::forward_training, dnnl::algorithm::pooling_max, src_md, dst_md,
        strides_dims, kernel_dims, dilation, padding_dims_l, padding_dims_r);
    auto pooling_pd = dnnl::pooling_v2_forward::primitive_desc(pooling_d, *onednn_engine);
    auto workspace_mem = dnnl::memory(pooling_pd.workspace_desc(), *onednn_engine);
    auto pooling_prim = dnnl::pooling_v2_forward(pooling_pd);

    pooling_prim.execute(
        *onednn_stream,
        {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}, {DNNL_ARG_WORKSPACE, workspace_mem}});
    onednn_stream->wait();
  }
};

}  // namespace oneflow
#endif  // WITH_ONEDNN
#endif  // ONEFLOW_USER_KERNELS_ONEDNN_POOL_KERNEL_UTIL_H_
