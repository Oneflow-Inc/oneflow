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
#include "oneflow/cambricon/cnnl/cnnl_op_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/ep/include/primitive/primitive.h"
#include "oneflow/core/ep/include/primitive/broadcast_matmul.h"
#include "oneflow/core/ep/common/primitive/broadcast_matmul.h"

namespace oneflow {
namespace ep {
namespace primitive {
namespace broadcast_matmul {
namespace internal {

namespace {

constexpr size_t kMaxNumDims = 8;

void LaunchBroadcastMatmul(Stream* stream, DataType data_type, BlasTransposeType transpose_a,
                           BlasTransposeType transpose_b, int64_t num_batch_dims,
                           const int64_t* broadcast_batch_dims, const int64_t* a_batch_dims,
                           const int64_t* b_batch_dims, const int64_t* c_batch_dims, int64_t m,
                           int64_t n, int64_t k, Scalar alpha, const void* a, const void* b,
                           Scalar beta, void* c) {
  auto* mlu_stream = stream->As<MluStream>();
  cnnlDataType_t cnnl_data_type = ConvertToCnnlDataType(data_type);
  int32_t is_trans_a = transpose_a == BlasTransposeType::T;
  int32_t is_trans_b = transpose_b == BlasTransposeType::T;

  CnnlMatmulDescriptor matmul_desc;
  matmul_desc.set_attr(CNNL_MATMUL_DESC_COMPUTE_TYPE, &cnnl_data_type, sizeof(cnnlDataType_t));
  matmul_desc.set_attr(CNNL_MATMUL_DESC_TRANSA, &is_trans_a, sizeof(int32_t));
  matmul_desc.set_attr(CNNL_MATMUL_DESC_TRANSB, &is_trans_b, sizeof(int32_t));

  float cnnl_alpha = alpha.Value<float>();
  float cnnl_beta = beta.Value<float>();

  cnnlMatMulAlgo_t algo;
  cnnlMatMulHeuristicResult_t result;
  OF_CNNL_CHECK(cnnlMatMulAlgoCreate(&algo));
  OF_CNNL_CHECK(cnnlCreateMatMulHeuristicResult(&result));

  if (num_batch_dims == 0) {
    CnnlTensorDescriptor a_desc, b_desc, c_desc;
    int64_t a_dims[2] = {is_trans_a ? k : m, is_trans_a ? m : k};
    int64_t b_dims[2] = {is_trans_b ? n : k, is_trans_b ? k : n};
    int64_t c_dims[2] = {m, n};
    a_desc.set(2, a_dims, cnnl_data_type);
    b_desc.set(2, b_dims, cnnl_data_type);
    c_desc.set(2, c_dims, cnnl_data_type);

    int return_algo_count = 0;
    OF_CNNL_CHECK(cnnlGetMatMulAlgoHeuristic(mlu_stream->cnnl_handle(), matmul_desc.desc(),
                                             a_desc.desc(), b_desc.desc(), c_desc.desc(),
                                             c_desc.desc(), NULL, 1, &result, &return_algo_count));
    size_t workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetMatMulHeuristicResult(result, algo, &workspace_size));
    CnnlWorkspace workspace(mlu_stream, workspace_size);
    // d = alpha * a * b + beta * c
    OF_CNNL_CHECK(cnnlMatMul_v2(mlu_stream->cnnl_handle(), matmul_desc.desc(), algo, &cnnl_alpha,
                                a_desc.desc(), a, b_desc.desc(), b, &cnnl_beta, c_desc.desc(), c,
                                workspace.dptr(), workspace_size, c_desc.desc(), c));
  } else {
    CnnlTensorDescriptor a_desc, b_desc, c_desc;
    std::vector<int64_t> a_dims(num_batch_dims + 2);
    std::vector<int64_t> b_dims(num_batch_dims + 2);
    std::vector<int64_t> c_dims(num_batch_dims + 2);
    for (int i = 0; i < num_batch_dims; ++i) {
      a_dims[i] = a_batch_dims[i];
      b_dims[i] = b_batch_dims[i];
      c_dims[i] = c_batch_dims[i];
    }
    a_dims[num_batch_dims] = is_trans_a ? k : m;
    a_dims[num_batch_dims + 1] = is_trans_a ? m : k;
    b_dims[num_batch_dims] = is_trans_b ? n : k;
    b_dims[num_batch_dims + 1] = is_trans_b ? k : n;
    c_dims[num_batch_dims] = m;
    c_dims[num_batch_dims + 1] = n;
    a_desc.set(a_dims.size(), a_dims.data(), cnnl_data_type);
    b_desc.set(b_dims.size(), b_dims.data(), cnnl_data_type);
    c_desc.set(c_dims.size(), c_dims.data(), cnnl_data_type);

    int return_algo_count = 0;
    OF_CNNL_CHECK(cnnlGetBatchMatMulAlgoHeuristic(mlu_stream->cnnl_handle(), matmul_desc.desc(),
                                                  a_desc.desc(), b_desc.desc(), c_desc.desc(), NULL,
                                                  1, &result, &return_algo_count));
    size_t workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetBatchMatMulHeuristicResult(result, algo, &workspace_size));
    CnnlWorkspace workspace(mlu_stream, workspace_size);
    // c = alpha * a * b + beta * c
    OF_CNNL_CHECK(cnnlBatchMatMulBCast_v2(
        mlu_stream->cnnl_handle(), matmul_desc.desc(), algo, &cnnl_alpha, a_desc.desc(), a,
        b_desc.desc(), b, &cnnl_beta, c_desc.desc(), c, workspace.dptr(), workspace_size));
  }

  // destory matmul result and algo handle
  OF_CNNL_CHECK(cnnlDestroyMatMulHeuristicResult(result));
  OF_CNNL_CHECK(cnnlMatMulAlgoDestroy(algo));
}

class BroadcastMatmulFactoryImpl : public BroadcastMatmulFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastMatmulFactoryImpl);
  BroadcastMatmulFactoryImpl() = default;
  ~BroadcastMatmulFactoryImpl() override = default;

  std::unique_ptr<BroadcastMatmul> New(DataType data_type, BlasTransposeType transpose_a,
                                       BlasTransposeType transpose_b,
                                       size_t max_num_dims) override {
    if (max_num_dims <= kMaxNumDims) {
      return std::make_unique<BroadcastMatmulImpl<kMaxNumDims>>(data_type, transpose_a,
                                                                transpose_b);
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kMLU, BroadcastMatmulFactory, BroadcastMatmulFactoryImpl);

}  // namespace

}  // namespace internal
}  // namespace broadcast_matmul
}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
