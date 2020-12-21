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
#include "oneflow/user/utils/unfold_util.h"
#include "oneflow/user/kernels/unfold_kernel_util.h"

namespace oneflow {

namespace user_op {

template<typename T>
class UnfoldKernelUtil<DeviceType::kCPU, T> {
 public:
  static void CFirstForward(const DeviceCtx* device_ctx, const Shape& in, const Shape& out_5d,
                            const Shape& out, const std::vector<int32_t>& kernel_size,
                            const std::vector<int32_t>& strides,
                            const std::vector<int32_t>& dilation_rate,
                            const std::vector<int32_t>& padding_before, const T* data_im,
                            T* data_col) {
    std::memset(data_col, T(0), out.elem_cnt() * sizeof(T));
    for (int64_t n = 0; n < in.At(0); ++n) {
      const T* data_im_batch = data_im + n * in.Count(1);
      T* data_col_batch = data_col + n * out.Count(1);
      for (int64_t c_col = 0; c_col < out.At(1); ++c_col) {
        const int64_t w_offset = c_col % kernel_size.at(2);
        const int64_t h_offset = (c_col / kernel_size.at(2)) % kernel_size.at(1);
        const int64_t d_offset =
            ((c_col / kernel_size.at(2)) / kernel_size.at(1)) % kernel_size.at(0);
        const int64_t c_im = c_col / kernel_size.at(0) / kernel_size.at(1) / kernel_size.at(2);

        for (int64_t d_col = 0; d_col < out_5d.At(2); ++d_col) {
          const int64_t d_im =
              d_col * strides.at(0) - padding_before.at(0) + d_offset * dilation_rate.at(0);

          for (int64_t h_col = 0; h_col < out_5d.At(3); ++h_col) {
            const int64_t h_im =
                h_col * strides.at(1) - padding_before.at(1) + h_offset * dilation_rate.at(1);

            for (int64_t w_col = 0; w_col < out_5d.At(4); ++w_col) {
              const int64_t w_im =
                  w_col * strides.at(2) - padding_before.at(2) + w_offset * dilation_rate.at(2);
              data_col_batch[((c_col * out_5d.At(2) + d_col) * out_5d.At(3) + h_col) * out_5d.At(4)
                             + w_col] =
                  (d_im >= 0 && h_im >= 0 && w_im >= 0 && d_im < in.At(2) && h_im < in.At(3)
                   && w_im < in.At(4))
                      ? data_im_batch[((c_im * in.At(2) + d_im) * in.At(3) + h_im) * in.At(4)
                                      + w_im]
                      : static_cast<T>(0);
            }
          }
        }
      }
    }
  }

  static void CFirstBackward(const DeviceCtx* device_ctx, const Shape& in, const Shape& out_5d,
                             const Shape& out, const std::vector<int32_t>& kernel_size,
                             const std::vector<int32_t>& strides,
                             const std::vector<int32_t>& dilation_rate,
                             const std::vector<int32_t>& padding_before, const T* data_col,
                             T* data_im) {
    std::memset(data_im, T(0), in.elem_cnt() * sizeof(T));
    for (int64_t n = 0; n < in.At(0); ++n) {
      T* data_im_batch = data_im + n * in.Count(1);
      const T* data_col_batch = data_col + n * out.Count(1);
      for (int64_t c_col = 0; c_col < out.At(1); ++c_col) {
        const int64_t d_offset = c_col % kernel_size.at(0);
        const int64_t h_offset = (c_col / kernel_size.at(0)) % kernel_size.at(1);
        const int64_t w_offset =
            ((c_col / kernel_size.at(0)) / kernel_size.at(1)) % kernel_size.at(2);
        const int64_t c_im = c_col / kernel_size.at(0) / kernel_size.at(1) / kernel_size.at(2);

        for (int64_t d_col = 0; d_col < out_5d.At(2); ++d_col) {
          const int64_t d_im =
              d_col * strides.at(0) - padding_before.at(0) + d_offset * dilation_rate.at(0);

          for (int64_t h_col = 0; h_col < out_5d.At(3); ++h_col) {
            const int64_t h_im =
                h_col * strides.at(1) - padding_before.at(1) + h_offset * dilation_rate.at(1);

            for (int64_t w_col = 0; w_col < out_5d.At(4); ++w_col) {
              const int64_t w_im =
                  w_col * strides.at(2) - padding_before.at(2) + w_offset * dilation_rate.at(2);

              if (d_im >= 0 && h_im >= 0 && w_im >= 0 && d_im < in.At(2) && h_im < in.At(3)
                  && w_im < in.At(4)) {
                data_im_batch[((c_im * in.At(2) + d_im) * in.At(3) + h_im) * in.At(4) + w_im] +=
                    data_col_batch[((c_col * out_5d.At(2) + d_col) * out_5d.At(3) + h_col)
                                       * out_5d.At(4)
                                   + w_col];
              }
            }
          }
        }
      }
    }
  }
};

INSTANTIATE_UNFOLD_KERNEL_UTIL(DeviceType::kCPU, float)
INSTANTIATE_UNFOLD_KERNEL_UTIL(DeviceType::kCPU, double)

}  // namespace user_op

}  // namespace oneflow
