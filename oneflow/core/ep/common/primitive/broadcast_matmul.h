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
#ifndef ONEFLOW_CORE_EP_COMMON_PRIMITIVE_BROADCAST_MATMUL_H_
#define ONEFLOW_CORE_EP_COMMON_PRIMITIVE_BROADCAST_MATMUL_H_

#include "oneflow/core/ep/include/primitive/broadcast_matmul.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

namespace ep {
namespace primitive {

namespace broadcast_matmul {

inline void Simplify(size_t num_a_dims, const int64_t* a_dims, size_t num_b_dims,
                     const int64_t* b_dims, size_t num_c_dims, const int64_t* c_dims,
                     BlasTransposeType transpose_a, BlasTransposeType transpose_b, int64_t* m,
                     int64_t* n, int64_t* k, int64_t* num_batch_dims, int64_t* broadcast_batch_dims,
                     int64_t* a_batch_dims, int64_t* b_batch_dims, int64_t* c_batch_dims) {
  CHECK_GE(num_a_dims, 2);
  CHECK_GE(num_b_dims, 2);
  CHECK_GE(num_c_dims, 2);
  if (transpose_a == BlasTransposeType::N) {
    *m = a_dims[num_a_dims - 2];
    *k = a_dims[num_a_dims - 1];
  } else if (transpose_a == BlasTransposeType::T) {
    *m = a_dims[num_a_dims - 1];
    *k = a_dims[num_a_dims - 2];
  } else {
    UNIMPLEMENTED();
  }
  CHECK_GT(*m, 0);
  CHECK_GT(*k, 0);
  if (transpose_b == BlasTransposeType::N) {
    CHECK_EQ(b_dims[num_b_dims - 2], *k);
    *n = b_dims[num_b_dims - 1];
  } else if (transpose_b == BlasTransposeType::T) {
    CHECK_EQ(b_dims[num_b_dims - 1], *k);
    *n = b_dims[num_b_dims - 2];
  } else {
    UNIMPLEMENTED();
  }
  CHECK_GT(*n, 0);
  CHECK_EQ(c_dims[num_c_dims - 2], *m);
  CHECK_EQ(c_dims[num_c_dims - 1], *n);
  const size_t num_max_batch_dims = std::max(std::max(num_a_dims, num_b_dims), num_c_dims) - 2;
  auto MakeGetBatchDim = [num_max_batch_dims](size_t num_dims, const int64_t* dims) {
    const int64_t num_batch_dims = num_dims - 2;
    const int64_t num_padding_dims = num_max_batch_dims - num_batch_dims;
    return [num_padding_dims, dims](size_t index) {
      return index < num_padding_dims ? 1 : dims[index - num_padding_dims];
    };
  };
  auto GetABatchDim = MakeGetBatchDim(num_a_dims, a_dims);
  auto GetBBatchDim = MakeGetBatchDim(num_b_dims, b_dims);
  auto GetCBatchDim = MakeGetBatchDim(num_c_dims, c_dims);
  *num_batch_dims = 0;
  bool prev_broadcast_a = false;
  bool prev_broadcast_b = false;
  bool prev_broadcast_c = false;
  for (int64_t i = 0; i < num_max_batch_dims; ++i) {
    const int64_t a_dim = GetABatchDim(i);
    const int64_t b_dim = GetBBatchDim(i);
    const int64_t c_dim = GetCBatchDim(i);
    const int64_t broadcast_dim = std::max(std::max(a_dim, b_dim), c_dim);
    CHECK_GT(broadcast_dim, 0);
    const bool broadcast_a = (a_dim == 1);
    const bool broadcast_b = (b_dim == 1);
    const bool broadcast_c = (c_dim == 1);
    CHECK((a_dim == broadcast_dim) || broadcast_a);
    CHECK((b_dim == broadcast_dim) || broadcast_b);
    CHECK((c_dim == broadcast_dim) || broadcast_c);
    if (broadcast_dim == 1) {
      continue;
    } else if (*num_batch_dims != 0
               && (prev_broadcast_a == broadcast_a && prev_broadcast_b == broadcast_b
                   && prev_broadcast_c == broadcast_c)) {
      a_batch_dims[*num_batch_dims - 1] *= a_dim;
      b_batch_dims[*num_batch_dims - 1] *= b_dim;
      c_batch_dims[*num_batch_dims - 1] *= c_dim;
      broadcast_batch_dims[*num_batch_dims - 1] *= broadcast_dim;
    } else {
      a_batch_dims[*num_batch_dims] = a_dim;
      b_batch_dims[*num_batch_dims] = b_dim;
      c_batch_dims[*num_batch_dims] = c_dim;
      broadcast_batch_dims[*num_batch_dims] = broadcast_dim;
      *num_batch_dims += 1;
      prev_broadcast_a = broadcast_a;
      prev_broadcast_b = broadcast_b;
      prev_broadcast_c = broadcast_c;
    }
  }
  if (*num_batch_dims >= 1 && a_batch_dims[*num_batch_dims - 1] != 1
      && b_batch_dims[*num_batch_dims - 1] == 1 && c_batch_dims[*num_batch_dims - 1] != 1
      && transpose_a == BlasTransposeType::N) {
    *m *= a_batch_dims[*num_batch_dims - 1];
    *num_batch_dims -= 1;
  }
}

template<size_t max_num_dims, typename Func>
void ForEachMatmul(DataType data_type, size_t m, size_t n, size_t k, Scalar beta,
                   size_t num_batch_dims, const int64_t* broadcast_batch_dims,
                   const int64_t* a_batch_dims, const int64_t* b_batch_dims,
                   const int64_t* c_batch_dims, const void* a, const void* b, void* c, Func func) {
  if (num_batch_dims == 0) {
    func(a, b, c, beta);
    return;
  }
  const size_t size_of_data_type = GetSizeOfDataType(data_type);
  const size_t stride_a = m * k * size_of_data_type;
  const size_t stride_b = k * n * size_of_data_type;
  const size_t stride_c = m * n * size_of_data_type;
  int64_t broadcast_batch_count = 1;
  for (int64_t i = 0; i < num_batch_dims; ++i) { broadcast_batch_count *= broadcast_batch_dims[i]; }
  NdIndexOffsetHelper<int64_t, max_num_dims> broadcast_index_helper(broadcast_batch_dims,
                                                                    num_batch_dims);
  NdIndexOffsetHelper<int64_t, max_num_dims> a_index_helper(a_batch_dims, num_batch_dims);
  NdIndexOffsetHelper<int64_t, max_num_dims> b_index_helper(b_batch_dims, num_batch_dims);
  NdIndexOffsetHelper<int64_t, max_num_dims> c_index_helper(c_batch_dims, num_batch_dims);
  int64_t a_batch_index[max_num_dims]{};
  int64_t b_batch_index[max_num_dims]{};
  int64_t c_batch_index[max_num_dims]{};
  int64_t broadcast_batch_index[max_num_dims]{};
  bool init_c = true;
  for (int64_t broadcast_batch_id = 0; broadcast_batch_id < broadcast_batch_count;
       ++broadcast_batch_id) {
    broadcast_index_helper.OffsetToNdIndex(broadcast_batch_id, broadcast_batch_index);
    for (int64_t i = 0; i < num_batch_dims; ++i) {
      if (a_batch_dims[i] == 1) {
        a_batch_index[i] = 0;
      } else {
        a_batch_index[i] = broadcast_batch_index[i];
      }
      if (b_batch_dims[i] == 1) {
        b_batch_index[i] = 0;
      } else {
        b_batch_index[i] = broadcast_batch_index[i];
      }
      if (c_batch_dims[i] == 1) {
        c_batch_index[i] = 0;
        if (broadcast_batch_index[i] != 0) { init_c = false; }
      } else {
        c_batch_index[i] = broadcast_batch_index[i];
      }
    }
    const int64_t a_batch_id = a_index_helper.NdIndexToOffset(a_batch_index);
    const int64_t b_batch_id = b_index_helper.NdIndexToOffset(b_batch_index);
    const int64_t c_batch_id = c_index_helper.NdIndexToOffset(c_batch_index);
    const void* a_ptr = static_cast<const unsigned char*>(a) + a_batch_id * stride_a;
    const void* b_ptr = static_cast<const unsigned char*>(b) + b_batch_id * stride_b;
    void* c_ptr = static_cast<unsigned char*>(c) + c_batch_id * stride_c;
    const Scalar batch_beta = init_c ? beta : Scalar(1);
    func(a_ptr, b_ptr, c_ptr, batch_beta);
  }
}

namespace internal {

namespace {

void LaunchBroadcastMatmul(Stream* stream, DataType data_type, BlasTransposeType transpose_a,
                           BlasTransposeType transpose_b, int64_t num_batch_dims,
                           const int64_t* broadcast_batch_dims, const int64_t* a_batch_dims,
                           const int64_t* b_batch_dims, const int64_t* c_batch_dims, int64_t m,
                           int64_t n, int64_t k, Scalar alpha, const void* a, const void* b,
                           Scalar beta, void* c);

template<size_t max_num_dims>
class BroadcastMatmulImpl : public BroadcastMatmul {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastMatmulImpl);
  BroadcastMatmulImpl(DataType data_type, BlasTransposeType transpose_a,
                      BlasTransposeType transpose_b)
      : data_type_(data_type), transpose_a_(transpose_a), transpose_b_(transpose_b) {}
  ~BroadcastMatmulImpl() override = default;

  void Launch(Stream* stream, Scalar alpha, size_t num_a_dims, const int64_t* a_dims, const void* a,
              size_t num_b_dims, const int64_t* b_dims, const void* b, Scalar beta,
              size_t num_c_dims, const int64_t* c_dims, void* c) override {
    CHECK_LE(num_a_dims, max_num_dims);
    CHECK_LE(num_b_dims, max_num_dims);
    CHECK_LE(num_c_dims, max_num_dims);
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
    int64_t num_batch_dims = 0;
    int64_t broadcast_batch_dims[max_num_dims]{};
    int64_t a_batch_dims[max_num_dims]{};
    int64_t b_batch_dims[max_num_dims]{};
    int64_t c_batch_dims[max_num_dims]{};
    Simplify(num_a_dims, a_dims, num_b_dims, b_dims, num_c_dims, c_dims, transpose_a_, transpose_b_,
             &m, &n, &k, &num_batch_dims, broadcast_batch_dims, a_batch_dims, b_batch_dims,
             c_batch_dims);
    LaunchBroadcastMatmul(stream, data_type_, transpose_a_, transpose_b_, num_batch_dims,
                          broadcast_batch_dims, a_batch_dims, b_batch_dims, c_batch_dims, m, n, k,
                          alpha, a, b, beta, c);
  }

 private:
  DataType data_type_;
  BlasTransposeType transpose_a_;
  BlasTransposeType transpose_b_;
};

}  // namespace

}  // namespace internal

}  // namespace broadcast_matmul

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_COMMON_PRIMITIVE_BROADCAST_MATMUL_H_
