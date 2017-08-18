#include "oneflow/core/operator/convolution_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void ConvolutionOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_convolution_conf());
  mut_op_conf() = op_conf;

  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollDataTmpBn("col_buf");

  EnrollModelBn("weight");
  if (GetBoolFromSpecialConf("has_bias_term")) {
    EnrollModelBn("bias");
    EnrollModelTmpBn("bias_multiplier");
  }
}

const PbMessage& ConvolutionOp::GetSpecialConf() const {
  return op_conf().convolution_conf();
}

void ConvolutionOp::InferShape4FwBlobs(
    std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) const {
  Shape* input_shape_ptr = GetShapePtr4BnInOp(SoleIbn());
  Shape* output_shape_ptr = GetShapePtr4BnInOp(SoleObn());
  Shape* colbuf_shape_ptr = GetShapePtr4BnInOp("col_buf");
  auto conv_conf = op_conf().convolution_conf();
  int64_t batch_size = input_shape_ptr->At(0);
  int64_t c_i = input_shape_ptr->At(1);

  int32_t out_num = GetInt32FromSpecialConf("out_num");
  if (policy == kModelParallel) {
    BalancedSplitter splitter(out_num, parallel_num);
    out_num = splitter.At(parallel_id).size();
  }
  int64_t c_o = out_num;

  int64_t kernel_size = 1;
  int64_t output_size = 1;
  std::vector<int64_t> output_shape_vec = {batch_size, c_o};
  std::vector<int64_t> weight_shape_vec = {c_o, c_i};

  int64_t h_len = (input_shape_ptr->At(2) + 2 * conv_conf.pad_h()
                   - conv_conf.kernel_size_h())
                      / conv_conf.stride_h()
                  + 1;
  output_shape_vec.push_back(h_len);
  int64_t w_len = (input_shape_ptr->At(3) + 2 * conv_conf.pad_w()
                   - conv_conf.kernel_size_w())
                      / conv_conf.stride_w()
                  + 1;
  output_shape_vec.push_back(w_len);
  weight_shape_vec.push_back(conv_conf.kernel_size_h());
  weight_shape_vec.push_back(conv_conf.kernel_size_w());
  kernel_size *= conv_conf.kernel_size_h();
  kernel_size *= conv_conf.kernel_size_w();
  output_size *= h_len;
  output_size *= w_len;

  *output_shape_ptr = Shape(output_shape_vec);
  CHECK_EQ(output_shape_ptr->NumAxes(), input_shape_ptr->NumAxes());
  *colbuf_shape_ptr = Shape({batch_size, output_size, c_i * kernel_size});
  Shape* weight = GetShapePtr4BnInOp("weight");
  *weight = Shape(weight_shape_vec);

  if (GetBoolFromSpecialConf("has_bias_term")) {
    Shape* bias = GetShapePtr4BnInOp("bias");
    Shape* biasmult_shape_ptr = GetShapePtr4BnInOp("bias_multiplier");
    *bias = Shape({c_o});
    *biasmult_shape_ptr = Shape({output_size});
  }
}

REGISTER_OP(OperatorConf::kConvolutionConf, ConvolutionOp);

}  // namespace oneflow
