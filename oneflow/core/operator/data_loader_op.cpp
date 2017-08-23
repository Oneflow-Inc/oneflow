//#include "oneflow/core/operator/data_loader_op.h"
//#include "oneflow/core/job/job_desc.h"
//
// namespace oneflow {
//
// void DataLoaderOp::InitFromOpConf(const OperatorConf& op_conf) {
//  CHECK(op_conf.has_data_loader_conf());
//  mut_op_conf() = op_conf;
//
//  EnrollOutputBn("feature", false);
//  EnrollOutputBn("label", false);
//}
//
// const PbMessage& DataLoaderOp::GetSpecialConf() const {
//  return op_conf().data_loader_conf();
//}
//
// void DataLoaderOp::InferBlobDesc4FwBlobs(
//    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
//    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) const {
//  // useful vars
//  int32_t piece_size = JobDesc::Singleton()->piece_size();
//  auto op_conf = static_cast<const DataLoaderOpConf*>(&GetSpecialConf());
//  // feature shape
//  Shape feature_shape_of_one_ins(op_conf->shape_of_one_feature_ins());
//  std::vector<int64_t> feature_shape = {piece_size};
//  feature_shape.insert(feature_shape.end(),
//                       feature_shape_of_one_ins.dim_vec().begin(),
//                       feature_shape_of_one_ins.dim_vec().end());
//  GetBlobDesc4BnInOp("feature")->mut_shape() = Shape(feature_shape);
//  // label shape
//  GetBlobDesc4BnInOp("label")->mut_shape() = Shape({piece_size});
//}
//
// REGISTER_OP(OperatorConf::kDataLoaderConf, DataLoaderOp);
//
//}  // namespace oneflow
