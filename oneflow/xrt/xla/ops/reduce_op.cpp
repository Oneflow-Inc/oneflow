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
#include "oneflow/xrt/xla/ops/op_context.h"
#include "oneflow/xrt/xla/ops/op_kernel.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

#include "oneflow/xrt/api.h"
#include "oneflow/xrt/xla/xla_helpers.h"

namespace oneflow {
namespace xrt {
namespace mola {

class ReduceOp : public XlaOpKernel {
 public:
  void Compile(XlaOpContext *ctx) override {
    std::vector<int32_t> axis = ctx->Attr<std::vector<int>>("axis");
    Shape in_shape = ctx->SoleInputShape();
    for (int i = 0; i < axis.size(); ++i) {
      if (axis[i] < 0) { axis[i] += in_shape.NumAxes(); }
    }

    xla::XlaOp input = ctx->SoleInput();
    if (axis.size() == 0) {
      std::vector<int64_t> dim_vec{1};
      dim_vec.insert(dim_vec.end(), in_shape.dim_vec().begin(), in_shape.dim_vec().end());
      input = Reshape(input, AsShape(dim_vec));
      axis.resize(in_shape.NumAxes());
      std::iota(axis.begin(), axis.end(), 1);
    }

    xla::XlaBuilder *builder = ctx->builder();
    DataType data_type = ctx->SoleInputType();
    xla::XlaOp output = xla::Reduce(input, InitValue(builder, data_type), Reduction(data_type),
                                    std::vector<long long>{axis.begin(), axis.end()});

    bool keep_dims = ctx->Attr<bool>("keepdims");
    if (keep_dims) {
      for (int i = 0; i < axis.size(); ++i) { in_shape.Set(axis[i], 1); }
      output = Reshape(output, in_shape);
    } else {
      // Reshape to 1-d array if output is scalar in order to
      // keep consistent with oneflow.
      if (axis.size() == in_shape.NumAxes()) { output = Reshape(output, Shape({1})); }
    }
    ctx->SetSoleOutput(output);
  }

  virtual xla::XlaOp InitValue(xla::XlaBuilder *builder, const DataType &data_type) = 0;
  virtual xla::XlaComputation Reduction(const DataType &data_type) = 0;
};

class ReduceSumOp : public ReduceOp {
 public:
  xla::XlaOp InitValue(xla::XlaBuilder *builder, const DataType &data_type) override {
    return Zero(builder, data_type);
  }

  xla::XlaComputation Reduction(const DataType &data_type) override {
    return CreateAddFunc(data_type);
  }
};

REGISTER_XLA_OP_KERNEL(ReduceSum, ReduceSumOp).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
