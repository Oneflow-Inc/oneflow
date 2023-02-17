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

#include "oneflow/core/ep/include/primitive/broadcast_elementwise_binary.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/ep/common//primitive/constant_pad.h"
#include "oneflow/core/ep/common/primitive/broadcast_elementwise_binary.h"
#include "oneflow/core/ep/cpu/primitive/binary_functor.h"
#include "oneflow/core/ep/cpu/primitive/type_seq.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ep/cpu/cpu_stream.h"
#include "oneflow/core/ep/cpu/cpu_device.h"
#include "oneflow/core/ep/common/primitive/util.h"
#include "oneflow/core/ep/common/onednn.h"

namespace oneflow {

namespace ep {
namespace primitive {
namespace broadcast_elementwise_binary {

namespace {

template<typename T>
T GetValue(Scalar value) {
  return value.Value<T>();
}

template<>
float16 GetValue<float16>(Scalar value) {
  return static_cast<float16>(GetValue<float>(value));
}

template<>
bfloat16 GetValue<bfloat16>(Scalar value) {
  return static_cast<bfloat16>(GetValue<float>(value));
}

template<BinaryOp binary_op, typename Src, typename Dst>
struct BinaryLhsScalarFunctor {
  BinaryLhsScalarFunctor(Src scalar, Scalar attr0, Scalar attr1)
      : scalar(scalar), functor(attr0, attr1) {}
  Dst operator()(Src src) const { return functor(scalar, src); }
  const Src scalar;
  BinaryFunctor<DeviceType::kCPU, binary_op, Src, Dst> functor;
};

template<BinaryOp binary_op, typename Src, typename Dst>
struct BinaryRhsScalarFunctor {
  BinaryRhsScalarFunctor(Src scalar, Scalar attr0, Scalar attr1)
      : scalar(scalar), functor(attr0, attr1) {}
  Dst operator()(Src src) const { return functor(src, scalar); }
  const Src scalar;
  BinaryFunctor<DeviceType::kCPU, binary_op, Src, Dst> functor;
};

template<BinaryOp binary_op, typename Src, typename Dst>
void LaunchElementwise(CpuStream* cpu_stream, size_t simplified_num_dims,
                       const int64_t* simplified_src0_dims, const Src* src0,
                       const int64_t* simplified_src1_dims, const Src* src1, Dst* dst, Scalar attr0,
                       Scalar attr1) {
  const int64_t elem_cnt = GetElementCount(simplified_num_dims, simplified_src0_dims);
  auto functor = BinaryFunctor<DeviceType::kCPU, binary_op, Src, Dst>(attr0, attr1);
  cpu_stream->ParallelFor(0, elem_cnt, [functor, src0, src1, dst](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; i++) { dst[i] = functor(src0[i], src1[i]); }
  });
}

template<BinaryOp binary_op, typename Src, typename Dst>
void LaunchBinaryLhsScalar(CpuStream* cpu_stream, Src src0_value, size_t src1_elem_cnt,
                           const Src* src1, Dst* dst, Scalar attr0, Scalar attr1) {
  auto functor = BinaryLhsScalarFunctor<binary_op, Src, Dst>(src0_value, attr0, attr1);
  cpu_stream->ParallelFor(0, src1_elem_cnt, [functor, src1, dst](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; i++) { dst[i] = functor(src1[i]); }
  });
}

template<BinaryOp binary_op, typename Src, typename Dst>
void LaunchBinaryRhsScalar(CpuStream* cpu_stream, Src src1_value, size_t src0_elem_cnt,
                           const Src* src0, Dst* dst, Scalar attr0, Scalar attr1) {
  auto functor = BinaryRhsScalarFunctor<binary_op, Src, Dst>(src1_value, attr0, attr1);
  cpu_stream->ParallelFor(0, src0_elem_cnt, [functor, src0, dst](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; i++) { dst[i] = functor(src0[i]); }
  });
}

template<BinaryOp binary_op, typename Src, typename Dst>
void LaunchRowWithMatrix(CpuStream* cpu_stream, const int64_t* simplified_src0_dims,
                         const Src* src0, const int64_t* simplified_src1_dims, const Src* src1,
                         Dst* dst, Scalar attr0, Scalar attr1) {
  int64_t rows = simplified_src1_dims[0];
  int64_t cols = simplified_src0_dims[1];
  auto functor = BinaryFunctor<DeviceType::kCPU, binary_op, Src, Dst>(attr0, attr1);
  cpu_stream->ParallelFor(
      0, rows,
      [functor, src0, src1, dst, cols](int64_t begin, int64_t end) {
        for (int64_t row_idx = begin; row_idx < end; row_idx++) {
          const Src* src1_row = src1 + row_idx * cols;
          Dst* dst_row = dst + row_idx * cols;
          for (int64_t col_idx = 0; col_idx < cols; col_idx++) {
            dst_row[col_idx] = functor(src0[col_idx], src1_row[col_idx]);
          }
        }
      },
      1);
}

template<BinaryOp binary_op, typename Src, typename Dst>
void LaunchMatrixWithRow(CpuStream* cpu_stream, const int64_t* simplified_src0_dims,
                         const Src* src0, const int64_t* simplified_src1_dims, const Src* src1,
                         Dst* dst, Scalar attr0, Scalar attr1) {
  int64_t rows = simplified_src0_dims[0];
  int64_t cols = simplified_src1_dims[1];
  auto functor = BinaryFunctor<DeviceType::kCPU, binary_op, Src, Dst>(attr0, attr1);
  cpu_stream->ParallelFor(
      0, rows,
      [functor, src0, src1, dst, cols](int64_t begin, int64_t end) {
        for (int64_t row_idx = begin; row_idx < end; row_idx++) {
          const Src* src0_row = src0 + row_idx * cols;
          Dst* dst_row = dst + row_idx * cols;
          for (int64_t col_idx = 0; col_idx < cols; col_idx++) {
            dst_row[col_idx] = functor(src0_row[col_idx], src1[col_idx]);
          }
        }
      },
      1);
}

template<BinaryOp binary_op, typename Src, typename Dst>
void LaunchColWithMatrix(CpuStream* cpu_stream, const int64_t* simplified_src0_dims,
                         const Src* src0, const int64_t* simplified_src1_dims, const Src* src1,
                         Dst* dst, Scalar attr0, Scalar attr1) {
  int64_t rows = simplified_src0_dims[0];
  int64_t cols = simplified_src1_dims[1];
  auto functor = BinaryFunctor<DeviceType::kCPU, binary_op, Src, Dst>(attr0, attr1);
  cpu_stream->ParallelFor(
      0, rows,
      [functor, src0, src1, dst, cols](int64_t begin, int64_t end) {
        for (int64_t row_idx = begin; row_idx < end; row_idx++) {
          const Src* src1_row = src1 + row_idx * cols;
          Dst* dst_row = dst + row_idx * cols;
          for (int64_t col_idx = 0; col_idx < cols; col_idx++) {
            dst_row[col_idx] = functor(src0[row_idx], src1_row[col_idx]);
          }
        }
      },
      1);
}

template<BinaryOp binary_op, typename Src, typename Dst>
void LaunchMatrixWithCol(CpuStream* cpu_stream, const int64_t* simplified_src0_dims,
                         const Src* src0, const int64_t* simplified_src1_dims, const Src* src1,
                         Dst* dst, Scalar attr0, Scalar attr1) {
  int64_t rows = simplified_src1_dims[0];
  int64_t cols = simplified_src0_dims[1];
  auto functor = BinaryFunctor<DeviceType::kCPU, binary_op, Src, Dst>(attr0, attr1);
  cpu_stream->ParallelFor(
      0, rows,
      [functor, src0, src1, dst, cols](int64_t begin, int64_t end) {
        for (int64_t row_idx = begin; row_idx < end; row_idx++) {
          const Src* src0_row = src0 + row_idx * cols;
          Dst* dst_row = dst + row_idx * cols;
          for (int64_t col_idx = 0; col_idx < cols; col_idx++) {
            dst_row[col_idx] = functor(src0_row[col_idx], src1[row_idx]);
          }
        }
      },
      1);
}

template<BinaryOp binary_op, typename Src, typename Dst, typename IndexType>
void LaunchGeneral(CpuStream* cpu_stream, size_t simplified_num_dims,
                   const int64_t* simplified_src0_dims, const Src* src0,
                   const int64_t* simplified_src1_dims, const Src* src1,
                   const int64_t* simplified_dst_dims, Dst* dst, int64_t dst_elem_cnt, Scalar attr0,
                   Scalar attr1) {
  auto functor = BinaryFunctor<DeviceType::kCPU, binary_op, Src, Dst>(attr0, attr1);
  cpu_stream->ParallelFor(
      0, dst_elem_cnt,
      [functor, src0, src1, dst, simplified_num_dims, simplified_src0_dims, simplified_src1_dims,
       simplified_dst_dims](int64_t begin, int64_t end) {
        auto src0_index_helper =
            NdIndexOffsetHelper<IndexType, kMaxNumDims>(simplified_src0_dims, simplified_num_dims);
        auto src1_index_helper =
            NdIndexOffsetHelper<IndexType, kMaxNumDims>(simplified_src1_dims, simplified_num_dims);
        auto dst_index_helper = OffsetToIndexCalculator<IndexType, kMaxNumDims>(
            simplified_dst_dims, simplified_num_dims);
        IndexType src0_index[kMaxNumDims];
        IndexType src1_index[kMaxNumDims];
        IndexType dst_index[kMaxNumDims];
        for (IndexType offset = begin; offset < end; offset++) {
          dst_index_helper.OffsetToNdIndex(offset, dst_index, simplified_num_dims);
          for (int i = 0; i < kMaxNumDims; i++) {
            if (i < simplified_num_dims) {
              src0_index[i] = (simplified_src0_dims[i] != 1) ? dst_index[i] : 0;
              src1_index[i] = (simplified_src1_dims[i] != 1) ? dst_index[i] : 0;
            } else {
              src0_index[i] = 0;
              src1_index[i] = 0;
            }
          }
          const IndexType src0_offset =
              src0_index_helper.NdIndexToOffset(src0_index, simplified_num_dims);
          const IndexType src1_offset =
              src1_index_helper.NdIndexToOffset(src1_index, simplified_num_dims);
          dst[offset] = functor(src0[src0_offset], src1[src1_offset]);
        }
      });
}

template<BinaryOp binary_op, typename Src, typename Dst>
void LaunchGeneralDispatchIndexType(CpuStream* cpu_stream, size_t simplified_num_dims,
                                    const int64_t* simplified_src0_dims, const Src* src0,
                                    const int64_t* simplified_src1_dims, const Src* src1,
                                    const int64_t* simplified_dst_dims, Dst* dst, Scalar attr0,
                                    Scalar attr1) {
  const int64_t dst_elem_cnt = GetElementCount(simplified_num_dims, simplified_dst_dims);
  if (dst_elem_cnt < (GetMaxVal<int32_t>() / 2)) {
    LaunchGeneral<binary_op, Src, Dst, int32_t>(
        cpu_stream, simplified_num_dims, simplified_src0_dims, src0, simplified_src1_dims, src1,
        simplified_dst_dims, dst, dst_elem_cnt, attr0, attr1);
  } else {
    LaunchGeneral<binary_op, Src, Dst, int64_t>(
        cpu_stream, simplified_num_dims, simplified_src0_dims, src0, simplified_src1_dims, src1,
        simplified_dst_dims, dst, dst_elem_cnt, attr0, attr1);
  }
}

template<BinaryOp binary_op, typename Src, typename Dst>
void DispatchLaunch(Stream* stream, size_t num_src0_dims, const int64_t* src0_dims, const Src* src0,
                    size_t num_src1_dims, const int64_t* src1_dims, const Src* src1, Dst* dst,
                    Scalar attr0, Scalar attr1) {
  auto* cpu_stream = stream->As<CpuStream>();
  size_t simplified_num_dims = 0;
  int64_t simplified_src0_dims[kMaxNumDims];
  int64_t simplified_src1_dims[kMaxNumDims];
  int64_t simplified_dst_dims[kMaxNumDims];
  SimplifyBroadcastDims<kMaxNumDims>(num_src0_dims, src0_dims, num_src1_dims, src1_dims,
                                     &simplified_num_dims, simplified_src0_dims,
                                     simplified_src1_dims, simplified_dst_dims);
  CheckInplace(simplified_num_dims, simplified_src0_dims, src0, simplified_dst_dims, dst);
  CheckInplace(simplified_num_dims, simplified_src1_dims, src1, simplified_dst_dims, dst);
  if (IsDimsEquals(simplified_num_dims, simplified_src0_dims, simplified_num_dims,
                   simplified_src1_dims)) {
    LaunchElementwise<binary_op, Src, Dst>(cpu_stream, simplified_num_dims, simplified_src0_dims,
                                           src0, simplified_src1_dims, src1, dst, attr0, attr1);
  } else {
    if (simplified_num_dims == 1 && simplified_src0_dims[0] == 1) {
      LaunchBinaryLhsScalar<binary_op, Src, Dst>(cpu_stream, *src0, simplified_src1_dims[0], src1,
                                                 dst, attr0, attr1);
    } else if (simplified_num_dims == 1 && simplified_src1_dims[0] == 1) {
      LaunchBinaryRhsScalar<binary_op, Src, Dst>(cpu_stream, *src1, simplified_src0_dims[0], src0,
                                                 dst, attr0, attr1);
    } else if (simplified_num_dims == 2 && simplified_src0_dims[0] == 1
               && simplified_src0_dims[1] == simplified_src1_dims[1]) {
      LaunchRowWithMatrix<binary_op, Src, Dst>(cpu_stream, simplified_src0_dims, src0,
                                               simplified_src1_dims, src1, dst, attr0, attr1);
    } else if (simplified_num_dims == 2 && simplified_src1_dims[0] == 1
               && simplified_src0_dims[1] == simplified_src1_dims[1]) {
      LaunchMatrixWithRow<binary_op, Src, Dst>(cpu_stream, simplified_src0_dims, src0,
                                               simplified_src1_dims, src1, dst, attr0, attr1);
    } else if (simplified_num_dims == 2 && simplified_src0_dims[1] == 1
               && simplified_src0_dims[0] == simplified_src1_dims[0]) {
      LaunchColWithMatrix<binary_op, Src, Dst>(cpu_stream, simplified_src0_dims, src0,
                                               simplified_src1_dims, src1, dst, attr0, attr1);
    } else if (simplified_num_dims == 2 && simplified_src1_dims[1] == 1
               && simplified_src0_dims[0] == simplified_src1_dims[0]) {
      LaunchMatrixWithCol<binary_op, Src, Dst>(cpu_stream, simplified_src0_dims, src0,
                                               simplified_src1_dims, src1, dst, attr0, attr1);
    } else {
      LaunchGeneralDispatchIndexType<binary_op, Src, Dst>(
          cpu_stream, simplified_num_dims, simplified_src0_dims, src0, simplified_src1_dims, src1,
          simplified_dst_dims, dst, attr0, attr1);
    }
  }
}

template<BinaryOp binary_op, typename Src, typename Dst>
class BroadcastElementwiseBinaryImpl : public BroadcastElementwiseBinary {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastElementwiseBinaryImpl);
  BroadcastElementwiseBinaryImpl(Scalar attr0, Scalar attr1) : attr0(attr0), attr1(attr1) {}
  ~BroadcastElementwiseBinaryImpl() override = default;

  void Launch(Stream* stream, Scalar src0, size_t num_src1_dims, const int64_t* src1_dims,
              const void* src1_ptr, void* dst_ptr) override {
    auto* cpu_stream = stream->As<CpuStream>();
    const size_t elem_cnt = GetElementCount(num_src1_dims, src1_dims);
    Dst* dst = reinterpret_cast<Dst*>(dst_ptr);
    const Src* src1 = reinterpret_cast<const Src*>(src1_ptr);
    LaunchBinaryLhsScalar<binary_op, Src, Dst>(cpu_stream, GetValue<Src>(src0), elem_cnt, src1, dst,
                                               attr0, attr1);
  }
  void Launch(Stream* stream, size_t num_src0_dims, const int64_t* src0_dims, const void* src0_ptr,
              Scalar src1, void* dst_ptr) override {
    auto* cpu_stream = stream->As<CpuStream>();
    const size_t elem_cnt = GetElementCount(num_src0_dims, src0_dims);
    Dst* dst = reinterpret_cast<Dst*>(dst_ptr);
    const Src* src0 = reinterpret_cast<const Src*>(src0_ptr);
    LaunchBinaryRhsScalar<binary_op, Src, Dst>(cpu_stream, GetValue<Src>(src1), elem_cnt, src0, dst,
                                               attr0, attr1);
  }
  void Launch(Stream* stream, size_t num_src0_dims, const int64_t* src0_dims, const void* src0,
              size_t num_src1_dims, const int64_t* src1_dims, const void* src1,
              void* dst) override {
    DispatchLaunch<binary_op, Src, Dst>(
        stream, num_src0_dims, src0_dims, reinterpret_cast<const Src*>(src0), num_src1_dims,
        src1_dims, reinterpret_cast<const Src*>(src1), reinterpret_cast<Dst*>(dst), attr0, attr1);
  }

 private:
  Scalar attr0, attr1;
};

template<BinaryOp binary_op, typename Src, typename Dst>
std::unique_ptr<BroadcastElementwiseBinary> NewBroadcastElementwiseBinary(Scalar attr0,
                                                                          Scalar attr1) {
  return std::unique_ptr<BroadcastElementwiseBinary>(
      new BroadcastElementwiseBinaryImpl<binary_op, Src, Dst>(attr0, attr1));
}

#define NDARRAY_BINARY_TYPE_SEQ \
  CPU_PRIMITIVE_BOOL_TYPE_SEQ   \
  CPU_PRIMITIVE_INT8_TYPE_SEQ   \
  CPU_PRIMITIVE_UINT8_TYPE_SEQ  \
  CPU_PRIMITIVE_INT32_TYPE_SEQ  \
  CPU_PRIMITIVE_INT64_TYPE_SEQ  \
  CPU_PRIMITIVE_FLOAT_TYPE_SEQ  \
  CPU_PRIMITIVE_DOUBLE_TYPE_SEQ \
  CPU_PRIMITIVE_FLOAT16_TYPE_SEQ

#ifdef WITH_ONEDNN

uint32_t OnednnFormatTagMap[kMaxNumDims] = {dnnl_a,     dnnl_ab,     dnnl_abc,     dnnl_abcd,
                                            dnnl_abcde, dnnl_abcdef, dnnl_abcdefg, dnnl_abcdefgh};

inline void OneDnnBroadcastDims(dnnl::memory::dims* src0, size_t num_src0_dims,
                                const int64_t* src0_dims, dnnl::memory::dims* src1,
                                size_t num_src1_dims, const int64_t* src1_dims,
                                dnnl::memory::dims& dst) {
  const int64_t num_dims = dst.size();
  const int64_t num_src0_padding_dims = num_dims - num_src0_dims;
  const int64_t num_src1_padding_dims = num_dims - num_src1_dims;
  for (int64_t i = 0; i < num_dims; i++) {
    int64_t src0_dim = i < num_src0_padding_dims ? 1 : src0_dims[i - num_src0_padding_dims];
    int64_t src1_dim = i < num_src1_padding_dims ? 1 : src1_dims[i - num_src1_padding_dims];
    CHECK((src0_dim == src1_dim || src0_dim == 1 || src1_dim == 1));
    (*src0)[i] = src0_dim;
    (*src1)[i] = src1_dim;
    dst[i] = std::max(src0_dim, src1_dim);
  }
}

template<typename T, dnnl::algorithm algorithm, dnnl::memory::data_type src_onednn,
         dnnl::memory::data_type dst_onednn>
class OneDnnBroadcastElementwiseBinaryImpl : public BroadcastElementwiseBinary {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OneDnnBroadcastElementwiseBinaryImpl);
  OneDnnBroadcastElementwiseBinaryImpl(Scalar attr0, Scalar attr1) : attr0(attr0), attr1(attr1) {}
  ~OneDnnBroadcastElementwiseBinaryImpl() override = default;

  void Launch(Stream* stream, Scalar src0, size_t num_src1_dims, const int64_t* src1_dims,
              const void* src1, void* dst) override {
    T scalar_val = GetValue<T>(src0);
    const int64_t src0_dims = 1;
    Launch(stream, 1, &src0_dims, &scalar_val, num_src1_dims, src1_dims, src1, dst);
  }
  void Launch(Stream* stream, size_t num_src0_dims, const int64_t* src0_dims, const void* src0,
              Scalar src1, void* dst) override {
    T scalar_val = GetValue<T>(src1);
    const int64_t src1_dims = 1;
    Launch(stream, num_src0_dims, src0_dims, src0, 1, &src1_dims, &scalar_val, dst);
  }
  void Launch(Stream* stream, size_t num_src0_dims, const int64_t* src0_dims, const void* src0,
              size_t num_src1_dims, const int64_t* src1_dims, const void* src1,
              void* dst) override {
    stream->As<CpuStream>()->onednn_executor()->Launch([&](dnnl::engine* onednn_engine,
                                                           dnnl::stream* onednn_stream) {
      // onednn do not optimize for 3d tensor in our experiments, so expand it
      // to 4d if needed.
      // Note that only onednn "internal" dims will be affected, the shape
      // of oneflow tensor (including the output tensor) will remain unchanged.
      size_t num_dims = std::max(std::max(num_src0_dims, num_src1_dims), static_cast<size_t>(4));
      dnnl::memory::dims src_0_dims(num_dims);
      dnnl::memory::dims src_1_dims(num_dims);
      dnnl::memory::dims dst_dims(num_dims);
      const void* onednn_src0 = nullptr;
      const void* onednn_src1 = nullptr;

      // OneDNN inplace operations only support src_0
      if (src1 == dst) {
        onednn_src0 = src1;
        onednn_src1 = src0;
        OneDnnBroadcastDims(&src_0_dims, num_src1_dims, src1_dims, &src_1_dims, num_src0_dims,
                            src0_dims, dst_dims);
      } else {
        onednn_src0 = src0;
        onednn_src1 = src1;
        OneDnnBroadcastDims(&src_0_dims, num_src0_dims, src0_dims, &src_1_dims, num_src1_dims,
                            src1_dims, dst_dims);
      }

      CheckInplace(num_dims, src_0_dims.data(), onednn_src0, dst_dims.data(), dst);
      CheckInplace(num_dims, src_1_dims.data(), onednn_src1, dst_dims.data(), dst);

      auto src_0_md = dnnl::memory::desc(
          src_0_dims, src_onednn,
          static_cast<dnnl::memory::format_tag>(OnednnFormatTagMap[num_dims - 1]));
      auto src_1_md = dnnl::memory::desc(
          src_1_dims, src_onednn,
          static_cast<dnnl::memory::format_tag>(OnednnFormatTagMap[num_dims - 1]));
      auto dst_md = dnnl::memory::desc(
          dst_dims, dst_onednn,
          static_cast<dnnl::memory::format_tag>(OnednnFormatTagMap[num_dims - 1]));

      auto src_0_mem = dnnl::memory(src_0_md, *onednn_engine, (void*)onednn_src0);
      auto src_1_mem = dnnl::memory(src_1_md, *onednn_engine, (void*)onednn_src1);
      auto dst_mem = dnnl::memory(dst_md, *onednn_engine, dst);

      auto binary_d = dnnl::binary::desc(algorithm, src_0_md, src_1_md, dst_md);
      auto binary_pd = dnnl::binary::primitive_desc(binary_d, *onednn_engine);
      auto binary_prim = dnnl::binary(binary_pd);

      binary_prim.execute(
          *onednn_stream,
          {{DNNL_ARG_SRC_0, src_0_mem}, {DNNL_ARG_SRC_1, src_1_mem}, {DNNL_ARG_DST, dst_mem}});
    });
  }

 private:
  Scalar attr0, attr1;
};

#define CPU_PRIMITIVE_BINARY_ONEDNN_TYPE_SEQ                               \
  OF_PP_MAKE_TUPLE_SEQ(dnnl::memory::data_type::u8, DataType::kBool, bool) \
  OF_PP_MAKE_TUPLE_SEQ(dnnl::memory::data_type::f32, DataType::kFloat, float)

// OneDNN binary op does not support s32
// CPU_PRIMITIVE_ONEDNN_INT32_TYPE_SEQ

#define CPU_PRIMITIVE_BINARY_ONEDNN_UNIMPLEMENTED_TYPE_SEQ \
  CPU_PRIMITIVE_FLOAT16_TYPE_SEQ                           \
  CPU_PRIMITIVE_DOUBLE_TYPE_SEQ                            \
  CPU_PRIMITIVE_INT8_TYPE_SEQ                              \
  CPU_PRIMITIVE_UINT8_TYPE_SEQ                             \
  CPU_PRIMITIVE_INT32_TYPE_SEQ                             \
  CPU_PRIMITIVE_INT64_TYPE_SEQ

#define BINARY_ONEDNN_ADD OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kAdd, dnnl::algorithm::binary_add)
#define BINARY_ONEDNN_SUB OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kSub, dnnl::algorithm::binary_sub)
#define BINARY_ONEDNN_MUL OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kMul, dnnl::algorithm::binary_mul)
#define BINARY_ONEDNN_DIV OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kDiv, dnnl::algorithm::binary_div)
#define BINARY_ONEDNN_MAX OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kMax, dnnl::algorithm::binary_max)
#define BINARY_ONEDNN_MIN OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kMin, dnnl::algorithm::binary_min)

#define BINARY_ONEDNN_EQ OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kEqual, dnnl::algorithm::binary_eq)
#define BINARY_ONEDNN_NE OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kNotEqual, dnnl::algorithm::binary_ne)
#define BINARY_ONEDNN_LT OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLessThan, dnnl::algorithm::binary_lt)
#define BINARY_ONEDNN_LE OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLessEqual, dnnl::algorithm::binary_le)
#define BINARY_ONEDNN_GT OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kGreaterThan, dnnl::algorithm::binary_gt)
#define BINARY_ONEDNN_GE OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kGreaterEqual, dnnl::algorithm::binary_ge)

#define BINARY_MATH_OP_ONEDNN_PAIR \
  BINARY_ONEDNN_ADD                \
  BINARY_ONEDNN_SUB                \
  BINARY_ONEDNN_MUL                \
  BINARY_ONEDNN_DIV                \
  BINARY_ONEDNN_MAX                \
  BINARY_ONEDNN_MIN

#define BINARY_LOGICAL_COMPARISION_OP_ONEDNN_PAIR \
  BINARY_ONEDNN_EQ                                \
  BINARY_ONEDNN_NE                                \
  BINARY_ONEDNN_LT                                \
  BINARY_ONEDNN_LE                                \
  BINARY_ONEDNN_GT                                \
  BINARY_ONEDNN_GE

#define BINARY_LOGICAL_COMPARISION_OP_ONEDNN_UNIMPLEMENTED \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLogicalAnd, AND)         \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLogicalOr, OR)           \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLogicalXor, XOR)

template<typename T, dnnl::algorithm algorithm, dnnl::memory::data_type src_onednn,
         dnnl::memory::data_type dst_onednn>
std::unique_ptr<BroadcastElementwiseBinary> NewOneDnnBroadcastElementwiseBinary(Scalar attr0,
                                                                                Scalar attr1) {
  return std::unique_ptr<BroadcastElementwiseBinary>(
      new OneDnnBroadcastElementwiseBinaryImpl<T, algorithm, src_onednn, dst_onednn>(attr0, attr1));
}

#define MAKE_NEW_ONEDNN_BROADCAST_ELEMENTWISE_BINARY_MATH_ENTRY(binary_op_pair, data_type_pair) \
  {std::make_tuple(OF_PP_PAIR_FIRST(binary_op_pair), OF_PP_PAIR_SECOND(data_type_pair),         \
                   OF_PP_PAIR_SECOND(data_type_pair)),                                          \
   NewOneDnnBroadcastElementwiseBinary<                                                         \
       OF_PP_PAIR_THIRD(data_type_pair), OF_PP_PAIR_SECOND(binary_op_pair),                     \
       OF_PP_PAIR_FIRST(data_type_pair), OF_PP_PAIR_FIRST(data_type_pair)>},

#define MAKE_NEW_ONEDNN_BROADCAST_ELEMENTWISE_BINARY_COMPARASION_AND_LOGICAL_ENTRY(         \
    binary_op_pair, src_data_type_pair, dst_data_type_pair)                                 \
  {std::make_tuple(OF_PP_PAIR_FIRST(binary_op_pair), OF_PP_PAIR_SECOND(src_data_type_pair), \
                   OF_PP_PAIR_SECOND(dst_data_type_pair)),                                  \
   NewOneDnnBroadcastElementwiseBinary<                                                     \
       OF_PP_PAIR_THIRD(src_data_type_pair), OF_PP_PAIR_SECOND(binary_op_pair),             \
       OF_PP_PAIR_FIRST(src_data_type_pair), OF_PP_PAIR_FIRST(dst_data_type_pair)>},

#endif  // WITH_ONEDNN

class BroadcastElementwiseBinaryFactoryImpl : public BroadcastElementwiseBinaryFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastElementwiseBinaryFactoryImpl);
  BroadcastElementwiseBinaryFactoryImpl() = default;
  ~BroadcastElementwiseBinaryFactoryImpl() override = default;

  std::unique_ptr<BroadcastElementwiseBinary> New(BinaryOp op, DataType src_type, DataType dst_type,
                                                  size_t max_num_dims) override {
    return New(op, src_type, dst_type, max_num_dims, Scalar(), Scalar());
  }

  std::unique_ptr<BroadcastElementwiseBinary> New(BinaryOp op, DataType src_type, DataType dst_type,
                                                  size_t max_num_dims, Scalar attr0) override {
    return New(op, src_type, dst_type, max_num_dims, attr0, Scalar());
  }

  std::unique_ptr<BroadcastElementwiseBinary> New(BinaryOp binary_op, DataType src_type,
                                                  DataType dst_type, size_t max_num_dims,
                                                  Scalar attr0, Scalar attr1) override {
    if (max_num_dims > kMaxNumDims) { return nullptr; }
#define MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_MATH_ENTRY(binary_op, data_type_pair) \
  {std::make_tuple(binary_op, OF_PP_PAIR_SECOND(data_type_pair),                    \
                   OF_PP_PAIR_SECOND(data_type_pair)),                              \
   NewBroadcastElementwiseBinary<binary_op, OF_PP_PAIR_FIRST(data_type_pair),       \
                                 OF_PP_PAIR_FIRST(data_type_pair)>},

#define MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_COMPARASION_AND_LOGICAL_ENTRY(      \
    binary_op, src_data_type_pair, dst_data_type_pair)                            \
  {std::make_tuple(binary_op, OF_PP_PAIR_SECOND(src_data_type_pair),              \
                   OF_PP_PAIR_SECOND(dst_data_type_pair)),                        \
   NewBroadcastElementwiseBinary<binary_op, OF_PP_PAIR_FIRST(src_data_type_pair), \
                                 OF_PP_PAIR_FIRST(dst_data_type_pair)>},

#define MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_ACTIVATION_GRAD_ENTRY(binary_op, data_type_pair) \
  {std::make_tuple(binary_op, OF_PP_PAIR_SECOND(data_type_pair),                               \
                   OF_PP_PAIR_SECOND(data_type_pair)),                                         \
   NewBroadcastElementwiseBinary<binary_op, OF_PP_PAIR_FIRST(data_type_pair),                  \
                                 OF_PP_PAIR_FIRST(data_type_pair)>},

    static const std::map<
        std::tuple<BinaryOp, DataType, DataType>,
        std::function<std::unique_ptr<BroadcastElementwiseBinary>(Scalar, Scalar)>>
        new_broadcast_elementwise_binary_handle{
            OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_MATH_ENTRY,
                                             BINARY_MATH_OP_SEQ, NDARRAY_BINARY_TYPE_SEQ)

                OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
                    MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_MATH_ENTRY, BINARY_BITWISE_OP_SEQ,
                    CPU_PRIMITIVE_INT_TYPE_SEQ CPU_PRIMITIVE_BOOL_TYPE_SEQ)

                    OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
                        MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_COMPARASION_AND_LOGICAL_ENTRY,
                        BINARY_LOGICAL_OP_SEQ BINARY_COMPARISION_OP_SEQ, NDARRAY_BINARY_TYPE_SEQ,
                        CPU_PRIMITIVE_BOOL_TYPE_SEQ)

                        OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
                            MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_ACTIVATION_GRAD_ENTRY,
                            BINARY_ACTIVATION_BACKWARD_OP_SEQ, CPU_PRIMITIVE_FLOATING_TYPE_SEQ)

                            OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
                                MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_MATH_ENTRY,
                                BINARY_MATH_BACKWARD_OP_SEQ, CPU_PRIMITIVE_FLOATING_TYPE_SEQ)};

#undef MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_COMPARASION_AND_LOGICAL_ENTRY
#undef MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_MATH_ENTRY

#ifdef WITH_ONEDNN
    static const std::map<
        std::tuple<BinaryOp, DataType, DataType>,
        std::function<std::unique_ptr<BroadcastElementwiseBinary>(Scalar, Scalar)>>
        new_broadcast_elementwise_binary_onednn_handle{
            // For oneDNN binary op
            OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
                MAKE_NEW_ONEDNN_BROADCAST_ELEMENTWISE_BINARY_MATH_ENTRY, BINARY_MATH_OP_ONEDNN_PAIR,
                CPU_PRIMITIVE_BINARY_ONEDNN_TYPE_SEQ)
            // For OneDnn comparasion binary op
            OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
                MAKE_NEW_ONEDNN_BROADCAST_ELEMENTWISE_BINARY_COMPARASION_AND_LOGICAL_ENTRY,
                BINARY_LOGICAL_COMPARISION_OP_ONEDNN_PAIR, CPU_PRIMITIVE_BINARY_ONEDNN_TYPE_SEQ,
                CPU_PRIMITIVE_ONEDNN_BOOl_TYPE_SEQ)};

#undef MAKE_NEW_ONEDNN_BROADCAST_ELEMENTWISE_BINARY_COMPARASION_AND_LOGICAL_ENTRY
#undef MAKE_NEW_ONEDNN_BROADCAST_ELEMENTWISE_BINARY_MATH_ENTRY
    if (OneDnnIsEnabled()) {
      const auto iter = new_broadcast_elementwise_binary_onednn_handle.find(
          std::make_tuple(binary_op, src_type, dst_type));
      if (iter != new_broadcast_elementwise_binary_onednn_handle.end()) {
        return iter->second(attr0, attr1);
      }
    }

#endif
    const auto iter = new_broadcast_elementwise_binary_handle.find(
        std::make_tuple(binary_op, src_type, dst_type));
    if (iter != new_broadcast_elementwise_binary_handle.end()) {
      return iter->second(attr0, attr1);
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, BroadcastElementwiseBinaryFactory,
                           BroadcastElementwiseBinaryFactoryImpl);

}  // namespace
}  // namespace broadcast_elementwise_binary
}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
