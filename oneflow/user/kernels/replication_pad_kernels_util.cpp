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
#include "oneflow/user/kernels/replication_pad_kernels_util.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {
namespace user_op {

template<typename IN_T>
struct ReplicationPad1dFunctor<DeviceType::kCPU, IN_T> final {
  void operator()(ep::Stream* stream, const IN_T* src, IN_T* dest,
                  const NdIndexOffsetHelper<int64_t, 3>& index_helper, const int64_t n_batch,
                  const int64_t n_channel, const int64_t y_width, const int64_t x_width,
                  const int64_t pad_left) {
    const int64_t dest_num = n_channel * y_width;
    const int64_t src_num = n_channel * x_width;
    const int64_t elem_num = n_batch * dest_num;
    DoReplicationPad1d<IN_T>(src, dest, index_helper, elem_num, src_num, dest_num, y_width, x_width,
                             pad_left);
  }
};

template<typename IN_T>
struct ReplicationPad1dGradFunctor<DeviceType::kCPU, IN_T> final {
  void operator()(ep::Stream* stream, const IN_T* src, IN_T* dest,
                  const NdIndexOffsetHelper<int64_t, 3>& index_helper, const int64_t n_batch,
                  const int64_t n_channel, const int64_t dy_width, const int64_t dx_width,
                  const int64_t pad_left) {
    const int64_t dest_num = n_channel * dx_width;
    const int64_t src_num = n_channel * dy_width;
    const int64_t elem_num = n_batch * src_num;
    DoReplicationPad1dGrad<IN_T>(src, dest, index_helper, elem_num, src_num, dest_num, dy_width,
                                 dx_width, pad_left);
  }
};

template<typename IN_T>
struct ReplicationPad2dFunctor<DeviceType::kCPU, IN_T> final {
  void operator()(ep::Stream* stream, const IN_T* src, IN_T* dest,
                  const NdIndexOffsetHelper<int64_t, 4>& index_helper, const int64_t n_batch,
                  const int64_t n_channel, const int64_t y_height, const int64_t y_width,
                  const int64_t x_height, const int64_t x_width, const int64_t pad_left,
                  const int64_t pad_top) {
    const int64_t dest_num = n_channel * y_height * y_width;
    const int64_t src_num = n_channel * x_height * x_width;
    const int64_t elem_num = n_batch * dest_num;
    DoReplicationPad2d<IN_T>(src, dest, index_helper, elem_num, src_num, dest_num, y_height,
                             y_width, x_height, x_width, pad_left, pad_top);
  }
};

template<typename IN_T>
struct ReplicationPad2dGradFunctor<DeviceType::kCPU, IN_T> final {
  void operator()(ep::Stream* stream, const IN_T* src, IN_T* dest,
                  const NdIndexOffsetHelper<int64_t, 4>& index_helper, const int64_t n_batch,
                  const int64_t n_channel, const int64_t dy_height, const int64_t dy_width,
                  const int64_t dx_height, const int64_t dx_width, const int64_t pad_left,
                  const int64_t pad_top) {
    const int64_t dest_num = n_channel * dx_height * dx_width;
    const int64_t src_num = n_channel * dy_height * dy_width;
    const int64_t elem_num = n_batch * src_num;
    DoReplicationPad2dGrad<IN_T>(src, dest, index_helper, elem_num, src_num, dest_num, dy_height,
                                 dy_width, dx_height, dx_width, pad_left, pad_top);
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_REPLICATION_PAD_FUNCTOR, (DeviceType::kCPU),
                                 PADDING_DATA_TYPE_CPU_SEQ);

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_REPLICATION_PAD_GRAD_FUNCTOR, (DeviceType::kCPU),
                                 PADDING_DATA_TYPE_CPU_SEQ);

}  // namespace user_op
}  // namespace oneflow
