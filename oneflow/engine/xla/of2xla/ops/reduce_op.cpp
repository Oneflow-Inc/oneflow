#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "oneflow/engine/xla/of2xla/xla_op_compiler_registry.h"
#include "oneflow/engine/xla/of2xla/xla_op_compiler.h"
#include "oneflow/engine/xla/of2xla/xla_op_context.h"

#include "oneflow/engine/xla/of2xla/xla_helpers.h"

namespace oneflow {
namespace mla {

class ReduceOp : public XlaOpCompiler {
 public:
  void Compile(XlaOpContext *ctx) override {
    bool keep_dims = ctx->GetAttr<bool>("keep_dims");
    CHECK(!keep_dims) << "Currently not support keep_dims option.";

    std::vector<int> axis = ctx->GetAttr<std::vector<int>>("axis");
    Shape in_shape = ctx->InputShape("in");
    for (int i = 0; i < axis.size(); ++i) {
      if (axis[i] < 0) {
        axis[i] += in_shape.NumAxes();
      }
    }

    xla::XlaOp input = ctx->Input("in");
    if (axis.size() == 0) {
      std::vector<int64_t> dim_vec{1};
      dim_vec.insert(dim_vec.end(), in_shape.dim_vec().begin(),
                     in_shape.dim_vec().end());
      input = Reshape(input, Shape(dim_vec));
      axis.resize(in_shape.NumAxes());
      std::iota(axis.begin(), axis.end(), 1);
    }

    xla::XlaBuilder *builder = ctx->builder();
    DataType data_type = ctx->InputType("in");
    ctx->SetOutput("out", xla::Reduce(
        input, InitValue(builder, data_type), Reduction(data_type),
        std::vector<long long>{axis.begin(), axis.end()}));
  }

  virtual xla::XlaOp InitValue(xla::XlaBuilder *builder,
                               const DataType &data_type) = 0;
  virtual xla::XlaComputation Reduction(const DataType &data_type) = 0;
};

class ReduceSumOp : public ReduceOp {
 public:
  xla::XlaOp InitValue(xla::XlaBuilder *builder,
                       const DataType &data_type) override {
    return Zero(builder, data_type);
  }

  xla::XlaComputation Reduction(const DataType &data_type) override {
    return CreateAddFunc(data_type);
  }
};

REGISTER_XLA_OP_COMPILER(ReduceSum, ReduceSumOp);

}  // namespace mla
}  // namespace oneflow
