#include "operator/convolution_op.h"
#include "glog/logging.h"
#include "operator/operator_manager.h"
#include "operator/op_util.h"

namespace oneflow {

void ConvolutionOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_convolution_conf());
  mut_op_conf() = op_conf;
  
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollDataTmpBn("col_buf");
  
  EnrollModelBn("weight");
  EnrollModelBn("bias");
  EnrollModelTmpBn("bias_multiplier");
}

std::string ConvolutionOp::GetValueFromPbOpConf(const std::string& k) const {
  return GetValueFromPbMessage(op_conf().convolution_conf(), k);
}

void ConvolutionOp::InferShape4FwBlobs(
    std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
    ParallelPolicy policy,
    uint64_t parallel_id,
    uint64_t parallel_num) const {
  TODO();
  /*
  Shape* input_shape_ptr = GetShapePtr4BnInOp(SoleIbn());
  Shape* output_shape_ptr = GetShapePtr4BnInOp(SoleObn());
  Shape* colbuf_shape_ptr = GetShapePtr4BnInOp(data_tmp_bns().at(0));
  auto conv_conf = op_conf().convolution_conf();
  auto pad = CheckDimPara4CnnOrPooling(conv_conf.pad(), 
                                       conv_conf.pad_h(), 
                                       conv_conf.pad_w());
  auto kernel = CheckDimPara4CnnOrPooling(conv_conf.kernel_size(), 
                                          conv_conf.kernel_h(),
                                          conv_conf.kernel_w());
  auto stride = CheckDimPara4CnnOrPooling(conv_conf.stride(), 
                                          conv_conf.stride_h(),
                                          conv_conf.stride_w());
  int64_t batch_size = input_shape_ptr->At(0);
  int64_t c_i = input_shape_ptr->At(1);
  int64_t h_i = input_shape_ptr->At(2);
  int64_t w_i = input_shape_ptr->At(3);
  int64_t c_o = conv_conf.out_num();
  int64_t h_o = (h_i + 2 * pad.first - kernel.first) / stride.first + 1;
  int64_t w_o = (w_i + 2 * pad.second - kernel.second) / stride.second + 1;
  *output_shape_ptr = Shape(({batch_size, c_o, h_o, w_o}));
  *colbuf_shape_ptr = Shape({batch_size, h_o * w_o, 
                             c_i * kernel.first * kernel.second}); 
  Shape* weight = GetShapePtr4BnInOp(model_bns().at(0));
  Shape* bias  = GetShapePtr4BnInOp(model_bns().at(1));
  Shape* biasmult_shape_ptr = GetShapePtr4BnInOp(model_tmp_bns().at(0));
  *weight = Shape({c_o, c_i * kernel.first * kernel.second});
  *bias = Shape({c_o});
  *biasmult_shape_ptr = Shape({h_o * w_o});
  */
}

REGISTER_OP(OperatorConf::kConvolutionConf, ConvolutionOp);

} // namespace oneflow
