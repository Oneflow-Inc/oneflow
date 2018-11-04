#include "oneflow/core/operator/bbox_nms_and_limit_op.h"

namespace oneflow {

void BboxNmsAndLimitOp::InitFromOpConf() {
  CHECK(op_conf().has_bbox_nms_and_limit_conf());
  EnrollInputBn("bbox", false);
  EnrollInputBn("bbox_pred", false);
  EnrollInputBn("bbox_prob", false);
  EnrollOutputBn("out_bbox", false);
  EnrollOutputBn("out_bbox_score", false);
  EnrollOutputBn("out_bbox_label", false);
  EnrollDataTmpBn("target_bbox");
  EnrollDataTmpBn("bbox_score");
}

const PbMessage& BboxNmsAndLimitOp::GetCustomizedConf() const {
  return op_conf().bbox_nms_and_limit_conf();
}

void BboxNmsAndLimitOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  kernel_conf->set_data_type(GetBlobDesc4BnInOp("bbox_pred")->data_type());
}

void BboxNmsAndLimitOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  CHECK_EQ(device_type(), DeviceType::kCPU);
  const BboxNmsAndLimitOpConf& conf = op_conf().bbox_nms_and_limit_conf();
  if (conf.bbox_vote_enabled()) { CHECK(conf.has_bbox_vote()); }
  int32_t img_record = Global<JobDesc>::Get()->DevicePieceSize4ParallelCtx(*parallel_ctx);
  int32_t num_limit = conf.detections_per_im() * img_record;
  // input: bbox (r, 5)
  const BlobDesc* bbox_bd = GetBlobDesc4BnInOp("bbox");
  // input: bbox_pred (r, c * 4)
  const BlobDesc* bbox_pred_bd = GetBlobDesc4BnInOp("bbox_pred");
  // input: bbox_prob (r, c)
  const BlobDesc* bbox_prob_bd = GetBlobDesc4BnInOp("bbox_prob");
  const int64_t num_boxes = bbox_bd->shape().At(0);
  const int64_t num_classes = bbox_prob_bd->shape().At(1);
  CHECK_EQ(bbox_bd->shape().At(1), 5);
  CHECK_EQ(bbox_pred_bd->shape().At(0), num_boxes);
  CHECK_EQ(bbox_prob_bd->shape().At(0), num_boxes);
  CHECK_EQ(bbox_pred_bd->shape().At(1), num_classes * 4);
  CHECK(!bbox_bd->has_data_id_field());
  CHECK(!bbox_pred_bd->has_data_id_field());
  CHECK(!bbox_prob_bd->has_data_id_field());

  // output: out_bbox (num_limit, 5)
  BlobDesc* out_bbox_bd = GetBlobDesc4BnInOp("out_bbox");
  out_bbox_bd->mut_shape() = Shape({num_limit, 5});
  out_bbox_bd->set_data_type(bbox_bd->data_type());
  out_bbox_bd->mut_dim0_inner_shape() = Shape({1, num_limit});
  out_bbox_bd->set_has_dim0_valid_num_field(true);
  out_bbox_bd->set_has_record_id_in_device_piece_field(
      bbox_bd->has_record_id_in_device_piece_field());
  // output: out_bbox_label (num_limit)
  BlobDesc* out_bbox_label_bd = GetBlobDesc4BnInOp("out_bbox_label");
  out_bbox_label_bd->mut_shape() = Shape({num_limit});
  out_bbox_label_bd->set_data_type(DataType::kInt32);
  out_bbox_label_bd->mut_dim0_inner_shape() = Shape({1, num_limit});
  out_bbox_label_bd->set_has_dim0_valid_num_field(true);
  out_bbox_label_bd->set_has_record_id_in_device_piece_field(
      bbox_bd->has_record_id_in_device_piece_field());
  // output: out_bbox_score (num_limit)
  BlobDesc* out_bbox_score_bd = GetBlobDesc4BnInOp("out_bbox_score");
  out_bbox_score_bd->mut_shape() = Shape({num_limit});
  out_bbox_score_bd->set_data_type(bbox_prob_bd->data_type());
  out_bbox_score_bd->mut_dim0_inner_shape() = Shape({1, num_limit});
  out_bbox_score_bd->set_has_dim0_valid_num_field(true);
  out_bbox_score_bd->set_has_record_id_in_device_piece_field(
      bbox_bd->has_record_id_in_device_piece_field());

  // datatmp: target_bbox (r, c, 5)
  BlobDesc* target_bbox_bd = GetBlobDesc4BnInOp("target_bbox");
  target_bbox_bd->mut_shape() = Shape({num_boxes, num_classes, 5});
  target_bbox_bd->set_data_type(bbox_bd->data_type());
  // datatmp: voting_score (r, c)
  BlobDesc* voting_score_bd = GetBlobDesc4BnInOp("bbox_score");
  voting_score_bd->mut_shape() = Shape({num_boxes, num_classes});
  voting_score_bd->set_data_type(bbox_prob_bd->data_type());
}

REGISTER_CPU_OP(OperatorConf::kBboxNmsAndLimitConf, BboxNmsAndLimitOp);

}  // namespace oneflow
