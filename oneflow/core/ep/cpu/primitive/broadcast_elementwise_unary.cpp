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

#include "oneflow/core/ep/include/primitive/broadcast_elementwise_unary.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/ep/common/primitive/broadcast_elementwise_unary.h"
#include "oneflow/core/ep/cpu/primitive/unary_functor.h"
#include "oneflow/core/ep/cpu/primitive/type_seq.h"
#include "oneflow/core/ep/cpu/cpu_stream.h"
#include "oneflow/core/ep/cpu/cpu_device.h"

namespace oneflow {

namespace ep {
namespace primitive {
namespace broadcast_elementwise_unary {

namespace {

template<UnaryOp unary_op, typename Src, typename Dst>
class BroadcastElementwiseUnaryImpl : public BroadcastElementwiseUnary {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastElementwiseUnaryImpl);
  BroadcastElementwiseUnaryImpl(Scalar attr0, Scalar attr1) : attr0(attr0), attr1(attr1) {}
  ~BroadcastElementwiseUnaryImpl() override = default;

  void Launch(Stream* stream, size_t num_src_dims, const int64_t* src_dims,
              const int64_t* src_strides, const Src* src, size_t num_dst_dims,
              const int64_t* dst_dims, const int64_t* dst_strides, Dst* dst) override {
    auto* cpu_stream = stream->As<CpuStream>();
    size_t simplified_num_dims = 0;
    int64_t simplified_src_dims[kMaxNumDims];
    int64_t simplified_dst_dims[kMaxNumDims];
    int64_t simplified_src_strides[kMaxNumDims];
    int64_t simplified_dst_strides[kMaxNumDims];
    SimplifyBroadcastDims<kMaxNumDims>(num_src_dims, src_dims, src_strides, num_dst_dims, dst_dims,
                                       dst_strides, &simplified_num_dims, simplified_src_dims,
                                       simplified_src_strides, simplified_dst_dims,
                                       simplified_dst_strides);
    CheckInplace(simplified_num_dims, simplified_src_dims, src, nullptr, nullptr,
                 simplified_dst_dims, dst);
    if (simplified_num_dims == 1 && simplified_src_dims[0] == 1) {
      auto functor = UnaryFunctor<DeviceType::kCPU, unary_op, Src, Dst>(attr0, attr1);
      Dst scalar_res = functor(*src);
      const int64_t elem_cnt = simplified_dst_dims[0];
      const int64_t dst_stride = simplified_dst_strides[0];
      cpu_stream->ParallelFor(0, elem_cnt, [dst, dst_stride](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; i++) { dst[i * dst_stride] = scalar_res; }
      });
    } else if (simplified_num_dims == 1) {  // 输入输出完全连续且无广播
      const int64_t elem_cnt = simplified_src_dims[0];
      const int64_t src_stride = simplified_src_strides[0];
      const int64_t dst_stride = simplified_dst_strides[0];
      auto functor = UnaryFunctor<DeviceType::kCPU, unary_op, Src, Dst>(attr0, attr1);
      cpu_stream->ParallelFor(
          0, elem_cnt, [functor, src, dst, src_stride, dst_stride](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; i++) {
              dst[i * dst_stride] = functor(src[i * src_stride]);
            }
          });
    } else {
      bool continuous_output = true;
      for (int i = simplified_num_dims - 1; i >= 0; i--) {
        if ((i == simplified_num_dims - 1 && simplified_dst_strides[i] != 1)
            || (i != simplified_num_dims - 1
                && simplified_dst_strides[i]
                       != simplified_dst_strides[i + 1] * simplified_dst_dims[i + 1])) {
          continuous_output = false;
          break;
        }
      }
      const int64_t elem_cnt = GetElementCount(simplified_num_dims, simplified_dst_dims);
      auto functor = UnaryFunctor<DeviceType::kCPU, unary_op, Src, Dst>(attr0, attr1);

      cpu_stream->ParallelFor(
          0, elem_cnt,
          [functor, src, dst, simplified_num_dims, simplified_src_dims, simplified_dst_dims,
           simplified_src_strides, simplified_dst_strides,
           continuous_output](int64_t begin, int64_t end) {
            auto src_index_to_offset_helper =
                IndexToOffsetWithStrideCalculator<int64_t, kMaxNumDims>(simplified_src_strides,
                                                                        simplified_num_dims);
            auto dst_offset_to_index_helper =
                OffsetToIndexWithStrideCalculator<int64_t, kMaxNumDims>(simplified_dst_dims,
                                                                        simplified_num_dims);
            auto dst_index_to_offset_helper =
                IndexToOffsetWithStrideCalculator<int64_t, kMaxNumDims>(simplified_dst_strides,
                                                                        simplified_num_dims);
            int64_t src_index[kMaxNumDims];
            int64_t dst_index[kMaxNumDims];
            for (int64_t offset = begin; offset < end; offset++) {
              dst_offset_to_index_helper.OffsetToNdIndex(offset, dst_index, simplified_num_dims);
              for (int i = 0; i < kMaxNumDims; i++) {
                if (i < simplified_num_dims) {
                  src_index[i] = (simplified_src_dims[i] != 1) ? dst_index[i] : 0;
                } else {
                  src_index[i] = 0;
                }
              }
              const int64_t src_offset =
                  src_index_to_offset_helper.NdIndexToOffset(src_index, simplified_num_dims);
              if (!continuous_output) {  // 输出完全连续
                const int64_t dst_offset =
                    dst_index_to_offset_helper.NdIndexToOffset(dst_index, simplified_num_dims);
                dst[dst_offset] = functor(src[src_offset]);
              } else {  // Naive 实现
                dst[offset] = functor(src[src_offset]);
              }
            }
          });
    }
  }

 protected:
  Scalar attr0, attr1;
};

template<UnaryOp unary_op, typename Src, typename Dst>
std::unique_ptr<BroadcastElementwiseUnary> NewBroadcastElementwiseUnary(Scalar attr0,
                                                                        Scalar attr1) {
  return std::unique_ptr<BroadcastElementwiseUnary>(
      new BroadcastElementwiseUnaryImpl<unary_op, Src, Dst>(attr0, attr1));
}

class BroadcastElementwiseUnaryFactoryImpl : public BroadcastElementwiseUnaryFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastElementwiseUnaryFactoryImpl);
  BroadcastElementwiseUnaryFactoryImpl() = default;
  ~BroadcastElementwiseUnaryFactoryImpl() override = default;

  std::unique_ptr<BroadcastElementwiseUnary> New(UnaryOp unary_op, DataType src_type,
                                                 DataType dst_type, size_t max_num_dims,
                                                 Scalar attr0, Scalar attr1) override {
    if (max_num_dims > kMaxNumDims) { return nullptr; }
#define MAKE_NEW_SAME_DTYPE_BROADCAST_ELEMENTWISE_UNARY_ENTRY(unary_op, dtype_pair)         \
  {std::make_tuple(unary_op, OF_PP_PAIR_SECOND(dtype_pair), OF_PP_PAIR_SECOND(dtype_pair)), \
   NewBroadcastElementwiseUnary<unary_op, OF_PP_PAIR_FIRST(dtype_pair),                     \
                                OF_PP_PAIR_FIRST(dtype_pair)>},

    static const std::map<std::tuple<UnaryOp, DataType, DataType>,
                          std::function<std::unique_ptr<BroadcastElementwiseUnary>(Scalar, Scalar)>>
        new_broadcast_elementwise_unary_handle{
            // TODO(yaozihang): add registry for ops which use BroadcastElementwiseUnary primitive
        };

#undef MAKE_NEW_SAME_DTYPE_BROADCAST_ELEMENTWISE_UNARY_ENTRY

    const auto iter =
        new_broadcast_elementwise_unary_handle.find(std::make_tuple(unary_op, src_type, dst_type));
    if (iter != new_broadcast_elementwise_unary_handle.end()) {
      return iter->second(attr0, attr1);
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, BroadcastElementwiseUnaryFactory,
                           BroadcastElementwiseUnaryFactoryImpl);

}  // namespace
}  // namespace broadcast_elementwise_unary
}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
