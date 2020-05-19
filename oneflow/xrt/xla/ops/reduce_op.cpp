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
    const auto& axis = ctx->Attr<std::vector<int>>("axis");
    Shape in_shape = ctx->InputShape("in");
    for (int i = 0; i < axis.size(); ++i) {
      if (axis[i] < 0) { axis[i] += in_shape.NumAxes(); }
    }

    xla::XlaOp input = ctx->Input("in");
    if (axis.size() == 0) {
      std::vector<int64_t> dim_vec{1};
      dim_vec.insert(dim_vec.end(), in_shape.dim_vec().begin(), in_shape.dim_vec().end());
      input = Reshape(input, AsShape(dim_vec));
      axis.resize(in_shape.NumAxes());
      std::iota(axis.begin(), axis.end(), 1);
    }

    xla::XlaBuilder *builder = ctx->builder();
    DataType data_type = ctx->InputType("in");
    xla::XlaOp output = xla::Reduce(input, InitValue(builder, data_type), Reduction(data_type),
                                    std::vector<long long>{axis.begin(), axis.end()});

    bool keep_dims = ctx->Attr<bool>("keep_dims");
    if (keep_dims) {
      for (int i = 0; i < axis.size(); ++i) { in_shape.Set(axis[i], 1); }
      output = Reshape(output, in_shape);
    } else {
      // Reshape to 1-d array if output is scalar in order to
      // keep consistent with oneflow.
      if (axis.size() == in_shape.NumAxes()) { output = Reshape(output, Shape({1})); }
    }
    ctx->SetOutput("out", output);
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
