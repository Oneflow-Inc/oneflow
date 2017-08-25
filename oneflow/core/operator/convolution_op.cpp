//#include "oneflow/core/operator/convolution_op.h"
//#include "oneflow/core/common/balanced_splitter.h"
//
// namespace oneflow {
//
// void ConvolutionOp::InitFromOpConf(const OperatorConf& op_conf) {
//  CHECK(op_conf.has_convolution_conf());
//  mut_op_conf() = op_conf;
//
//  EnrollInputBn("in");
//  EnrollOutputBn("out");
//  EnrollDataTmpBn("col_buf");
//
//  EnrollModelBn("weight");
//  if (GetBoolFromSpecialConf("has_bias_term")) {
//    EnrollModelBn("bias");
//    EnrollModelTmpBn("bias_multiplier");
//  }
//}
//
// const PbMessage& ConvolutionOp::GetSpecialConf() const {
//  return op_conf().convolution_conf();
//}
//
// void ConvolutionOp::InferBlobDesc4FwBlobs(
//    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
//    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) const {
//  const Shape& input_shape = GetBlobDesc4BnInOp(SoleIbn())->shape();
//  auto conv_conf = op_conf().convolution_conf();
//  int64_t batch_size = input_shape.At(0);
//  int64_t c_i = input_shape.At(1);
//
//  int32_t out_num = GetInt32FromSpecialConf("out_num");
//  if (policy == kModelParallel) {
//    BalancedSplitter splitter(out_num, parallel_num);
//    out_num = splitter.At(parallel_id).size();
//  }
//  int64_t c_o = out_num;
//
//  int64_t kernel_size = 1;
//  int64_t output_size = 1;
//  std::vector<int64_t> output_shape_vec = {batch_size, c_o};
//
//  int64_t h_len =
//      (input_shape.At(2) + 2 * conv_conf.pad_h() - conv_conf.kernel_size_h())
//          / conv_conf.stride_h()
//      + 1;
//  output_shape_vec.push_back(h_len);
//  int64_t w_len =
//      (input_shape.At(3) + 2 * conv_conf.pad_w() - conv_conf.kernel_size_w())
//          / conv_conf.stride_w()
//      + 1;
//  output_shape_vec.push_back(w_len);
//  kernel_size *= conv_conf.kernel_size_h();
//  kernel_size *= conv_conf.kernel_size_w();
//  output_size *= h_len;
//  output_size *= w_len;
//
//  GetBlobDesc4BnInOp(SoleObn())->mut_shape() = Shape(output_shape_vec);
//  GetBlobDesc4BnInOp("col_buf")->mut_shape() =
//      Shape({batch_size, output_size, c_i * kernel_size});
//  GetBlobDesc4BnInOp("weight")->mut_shape() = Shape({c_o, c_i * kernel_size});
//
//  if (GetBoolFromSpecialConf("has_bias_term")) {
//    GetBlobDesc4BnInOp("bias")->mut_shape() = Shape({c_o});
//    GetBlobDesc4BnInOp("bias_multiplier")->mut_shape() = Shape({output_size});
//  }
//}
//
// REGISTER_OP(OperatorConf::kConvolutionConf, ConvolutionOp);
//
//}  // namespace oneflow
