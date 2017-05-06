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

const PbMessage& ConvolutionOp::GetSpecialConf() const {
  return op_conf().convolution_conf();
}

void ConvolutionOp::InferShape4FwBlobs(
    std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
    ParallelPolicy policy,
    uint64_t parallel_id,
    uint64_t parallel_num) const {
  Shape* input_shape_ptr = GetShapePtr4BnInOp(SoleIbn());
  Shape* output_shape_ptr = GetShapePtr4BnInOp(SoleObn());
  Shape* colbuf_shape_ptr = GetShapePtr4BnInOp(data_tmp_bns().at(0));
  auto conv_conf = op_conf().convolution_conf();
  int64_t batch_size = input_shape_ptr->At(0);
  int64_t c_i = input_shape_ptr->At(1);
  int64_t c_o = conv_conf.out_num();
  int64_t kernel_size = 1;
  int64_t output_size = 1;
  std::vector<int64_t> output_shape_vec = {batch_size, c_o};
  for (int64_t i = 0; i < input_shape_ptr->NumAxes() - 2; ++i) {
    int64_t len = (input_shape_ptr->At(i + 2) + 2 * conv_conf.pad(i) - 
                  conv_conf.kernel_size(i)) / conv_conf.stride(i) + 1; 
    output_shape_vec.push_back(len);
    kernel_size *= conv_conf.kernel_size(i);
    output_size *= len;
  }
  *output_shape_ptr = Shape(output_shape_vec);
  *colbuf_shape_ptr = Shape({batch_size, output_size, c_i * kernel_size}); 
  Shape* weight = GetShapePtr4BnInOp(model_bns().at(0));
  Shape* bias  = GetShapePtr4BnInOp(model_bns().at(1));
  Shape* biasmult_shape_ptr = GetShapePtr4BnInOp(model_tmp_bns().at(0));
  *weight = Shape({1, c_o, c_i * kernel_size});
  *bias = Shape({1, c_o});
  *biasmult_shape_ptr = Shape({output_size});
}

REGISTER_OP(OperatorConf::kConvolutionConf, ConvolutionOp);

} // namespace oneflow
