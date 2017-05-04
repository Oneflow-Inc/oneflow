#include "operator/pooling_op.h"
#include <vector>
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

const PbMessage& PoolingOp::GetSpecialConf() const {
  return op_conf().pooling_conf();
}

void PoolingOp::InferShape4FwBlobs(
    std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
    ParallelPolicy policy,
    uint64_t parallel_id,
    uint64_t parallel_num) const {
  Shape* output_shape_ptr = GetShapePtr4BnInOp(SoleObn());
  Shape* input_shape_ptr = GetShapePtr4BnInOp(SoleIbn());
  const PoolingOpConf& pooling_conf = op_conf().pooling_conf();
  auto pad_vec = pooling_conf.pad();
  auto kernel_vec = pooling_conf.kernel_size();
  auto stride_vec = pooling_conf.stride();
  std::vector<int64_t> output_shape_dim_vec = {
      input_shape_ptr->At(0),
      input_shape_ptr->At(1)};
  CHECK_EQ(input_shape_ptr->NumAxes() - 2, pad_vec.size());
  CHECK_EQ(pad_vec.size(), kernel_vec.size());
  CHECK_EQ(pad_vec.size(), stride_vec.size());
  for (size_t i = 0; i < pad_vec.size(); i++) {
    int64_t temp = input_shape_ptr->At(i + 2) + 2 * pad_vec[i] - kernel_vec[i];
    output_shape_dim_vec.push_back(temp / stride_vec[i] + 1);
  }
  *output_shape_ptr = Shape(output_shape_dim_vec);
  CHECK_EQ(data_tmp_bns().size(), 1);
  Shape* data_tmp_shape_ptr = GetShapePtr4BnInOp(*(data_tmp_bns().begin()));
  *data_tmp_shape_ptr = Shape(output_shape_dim_vec);
}

REGISTER_OP(OperatorConf::kPoolingConf, PoolingOp);

}  // namespace oneflow
