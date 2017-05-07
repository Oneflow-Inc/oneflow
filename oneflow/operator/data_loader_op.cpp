#include "operator/data_loader_op.h"
#include "glog/logging.h"
#include "operator/operator_manager.h"
#include "job/job_desc.h"

namespace oneflow {

void DataLoaderOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_data_loader_conf());
  mut_op_conf() = op_conf;
 
  EnrollOutputBn("feature", false);
  EnrollOutputBn("label", false);
}

const PbMessage& DataLoaderOp::GetSpecialConf() const {
  return op_conf().data_loader_conf();
}

void DataLoaderOp::InferShape4FwBlobs(
    std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
    ParallelPolicy policy,
    uint64_t parallel_id,
    uint64_t parallel_num) const {
  // useful vars
  uint32_t piece_size = JobDesc::Singleton().piece_size();
  auto op_conf = of_dynamic_cast<const DataLoaderOpConf*> (&GetSpecialConf());
  // feature shape
  Shape feature_shape_of_one_ins(op_conf->shape_of_one_feature_ins());
  std::vector<int64_t> feature_shape = {piece_size};
  feature_shape.insert(feature_shape.end(),
                       feature_shape_of_one_ins.dim_vec().begin(),
                       feature_shape_of_one_ins.dim_vec().end());
  *GetShapePtr4BnInOp("feature") = Shape(feature_shape);
  // label shape
  *GetShapePtr4BnInOp("label") = Shape({piece_size, 1});
}

REGISTER_OP(OperatorConf::kDataLoaderConf, DataLoaderOp);

} // namespace oneflow
