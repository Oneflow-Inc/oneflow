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
#include "oneflow/core/primitive/include/broadcast_elementwise_binary.h"
#include "oneflow/core/primitive/common/broadcast_elementwise_binary.h"
#include "oneflow/core/primitive/cuda/type_seq.h"
#include "oneflow/core/stream/cuda_stream_context.h"

namespace oneflow {

namespace primitive {

namespace {

template<BinaryOp op, typename T, typename R>
struct BinaryFunctor {
  __device__ R operator()(T src0, T src1);
};

template<typename T, typename R>
struct BinaryFunctor<BinaryOp::kAdd, T, R> {
  __device__ R operator()(T src0, T src1) { return static_cast<R>(src0 + src1); }
};

template<BinaryOp op, typename T, typename R, size_t num_dims, size_t pack_size, typename IndexType>
__global__ void BroadcastElementwiseBinaryGpu(
    BroadcastElementwiseBinaryParams<num_dims, IndexType> params) {
  const PackType<T, pack_size>* src0 = reinterpret_cast<const PackType<T, pack_size>*>(params.src0);
  const PackType<T, pack_size>* src1 = reinterpret_cast<const PackType<T, pack_size>*>(params.src1);
  PackType<R, pack_size>* dst = reinterpret_cast<PackType<R, pack_size>*>(params.dst);
  IndexType src0_index[num_dims];
  IndexType src1_index[num_dims];
  IndexType dst_index[num_dims];
  CUDA_1D_KERNEL_LOOP_T(IndexType, offset, params.count) {
    params.dst_index_helper.OffsetToNdIndex(offset, dst_index);
    for (int64_t i = 0; i < num_dims; ++i) {
      if (params.src0_dims[i] == 1) {
        src0_index[i] = 0;
      } else {
        src0_index[i] = dst_index[i];
      }
      if (params.src1_dims[i] == 1) {
        src1_index[i] = 0;
      } else {
        src1_index[i] = dst_index[i];
      }
    }
    const IndexType src0_offset = params.src0_index_helper.NdIndexToOffset(src0_index);
    const IndexType src1_offset = params.src1_index_helper.NdIndexToOffset(src1_index);
    Pack<T, pack_size> src0_pack;
    src0_pack.storage = src0[src0_offset];
    Pack<T, pack_size> src1_pack;
    src1_pack.storage = src1[src1_offset];
    Pack<R, pack_size> dst_pack;
#pragma unroll
    for (int j = 0; j < pack_size; ++j) {
      dst_pack.elem[j] = BinaryFunctor<op, T, R>()(src0_pack.elem[j], src1_pack.elem[j]);
    }
    dst[offset] = dst_pack.storage;
  }
}

template<BinaryOp op, typename T, typename R, size_t num_dims, size_t pack_size, typename IndexType>
void LaunchKernel(StreamContext* stream_ctx,
                  BroadcastElementwiseBinaryParams<num_dims, IndexType> params) {
  cudaStream_t cuda_stream =
      CHECK_NOTNULL(dynamic_cast<CudaStreamContext*>(stream_ctx))->cuda_stream();
  BroadcastElementwiseBinaryGpu<op, T, R, num_dims, pack_size, IndexType>
      <<<BlocksNum4ThreadsNum(params.count), kCudaThreadsNumPerBlock, 0, cuda_stream>>>(params);
}

template<BinaryOp op, typename T, typename R>
class BroadcastElementwiseBinaryImpl : public BroadcastElementwiseBinary {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastElementwiseBinaryImpl);
  BroadcastElementwiseBinaryImpl() = default;
  ~BroadcastElementwiseBinaryImpl() override = default;

  void Launch(StreamContext* stream_ctx, Scalar src0, size_t num_src1_dims,
              const int64_t* src1_dims, const void* src1, void* dst) override {
    UNIMPLEMENTED();
  }
  void Launch(StreamContext* stream_ctx, size_t num_src0_dims, const int64_t* src0_dims,
              const void* src0, Scalar src1, void* dst) override {
    UNIMPLEMENTED();
  }
  void Launch(StreamContext* stream_ctx, size_t num_src0_dims, const int64_t* src0_dims,
              const void* src0, size_t num_src1_dims, const int64_t* src1_dims, const void* src1,
              void* dst) override {
    LOG(ERROR) << "BroadcastElementwiseBinaryImpl launch " << op;
    SimplifyThenLaunch<op, T, R>(stream_ctx, num_src0_dims, src0_dims, src0, num_src1_dims,
                                 src1_dims, src1, dst);
  }
};

template<BinaryOp op, typename T, typename R>
std::unique_ptr<BroadcastElementwiseBinary> NewBroadcastElementwiseBinary() {
  return std::unique_ptr<BroadcastElementwiseBinary>(
      new BroadcastElementwiseBinaryImpl<op, T, R>());
}

class BroadcastElementwiseBinaryFactoryImpl : public BroadcastElementwiseBinaryFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastElementwiseBinaryFactoryImpl);
  BroadcastElementwiseBinaryFactoryImpl() = default;
  ~BroadcastElementwiseBinaryFactoryImpl() override = default;

  std::unique_ptr<BroadcastElementwiseBinary> New(BinaryOp op, DataType src_type, DataType dst_type,
                                                  size_t max_num_dims) override {
    if (max_num_dims > kMaxNumDims) { return nullptr; }
#define MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_SAME_DTYPE_ENTRY(binary_op, data_type_pair) \
  {std::make_tuple(binary_op, OF_PP_PAIR_SECOND(data_type_pair),                          \
                   OF_PP_PAIR_SECOND(data_type_pair)),                                    \
   NewBroadcastElementwiseBinary<binary_op, OF_PP_PAIR_FIRST(data_type_pair),             \
                                 OF_PP_PAIR_FIRST(data_type_pair)>},

    static const std::map<std::tuple<BinaryOp, DataType, DataType>,
                          std::function<std::unique_ptr<BroadcastElementwiseBinary>()>>
        new_broadcast_elementwise_binary_handle{OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
            MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_SAME_DTYPE_ENTRY, CUDA_PRIMITIVE_BINARY_OP_SEQ,
            CUDA_PRIMITIVE_ALL_TYPE_SEQ)};

#undef MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_SAME_DTYPE_ENTRY

    const auto it =
        new_broadcast_elementwise_binary_handle.find(std::make_tuple(op, src_type, dst_type));
    if (it != new_broadcast_elementwise_binary_handle.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kGPU, BroadcastElementwiseBinaryFactory,
                           BroadcastElementwiseBinaryFactoryImpl);

}  // namespace

}  // namespace primitive

}  // namespace oneflow
