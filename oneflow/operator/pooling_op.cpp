#include "operator/pooling_op.h"
#include "glog/logging.h"
#include "operator/operator_manager.h"
#include "operator/op_util.h"

namespace oneflow {

void PoolingOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_pooling_conf());
  mut_op_conf() = op_conf;

  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollDataTmpBn("idx");
}

std::string PoolingOp::GetValueFromPbOpConf(const std::string& k) const {
  return GetValueFromPbMessage(op_conf().pooling_conf(), k);
}

REGISTER_OP(OperatorConf::kPoolingConf, PoolingOp);

void PoolingOp::InferShape4FwBlobs(
    std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
    ParallelPolicy policy,
    uint64_t parallel_id,
    uint64_t parallel_size) const {
  Shape* output_shape_ptr = GetShapePtr4BnInOp(SoleObn());
  Shape* input_shape_ptr = GetShapePtr4BnInOp(SoleIbn());
  const PoolingOpConf& pooling_conf = op_conf().pooling_conf();
  auto pad_pair = CheckDimPara(pooling_conf.pad(),
                               pooling_conf.pad_h(), pooling_conf.pad_w());
  uint32_t pad_h = pad_pair.first;
  uint32_t pad_w = pad_pair.second;
  auto kernel_pair = CheckDimPara(pooling_conf.kernel_size(),
                                  pooling_conf.kernel_h(), 
                                  pooling_conf.kernel_w());
  uint32_t kernel_h = kernel_pair.first;
  uint32_t kernel_w = kernel_pair.second;
  auto stride_pair = CheckDimPara(pooling_conf.stride(),
                                  pooling_conf.stride_h(), 
                                  pooling_conf.stride_w());
  uint32_t stride_h = stride_pair.first;
  uint32_t stride_w = stride_pair.second;
  // the input shape must be NxCxHxW
  CHECK_EQ(input_shape_ptr->NumAxes(), 4);
  std::vector<int64_t> output_shape_dim_vec;
  output_shape_dim_vec.push_back(input_shape_ptr->At(0));
  output_shape_dim_vec.push_back(input_shape_ptr->At(1));
  output_shape_dim_vec.push_back(
      (input_shape_ptr->At(2) + 2 * pad_h - kernel_h) / stride_h + 1 );
  output_shape_dim_vec.push_back(
      (input_shape_ptr->At(3) + 2 * pad_w - kernel_w) / stride_w + 1);
  *output_shape_ptr = Shape(output_shape_dim_vec);
  CHECK_EQ(data_tmp_bns().size(), 1);
  Shape* data_tmp_shape_ptr = GetShapePtr4BnInOp(*(data_tmp_bns().begin()));
  *data_tmp_shape_ptr = Shape(output_shape_dim_vec);
}

} // namespace oneflow
