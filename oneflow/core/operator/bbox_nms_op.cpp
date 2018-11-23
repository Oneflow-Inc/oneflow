#include "oneflow/core/operator/bbox_nms_op.h"

namespace oneflow {

void BboxNmsOp::InitFromOpConf() {
  CHECK(op_conf().has_bbox_nms_conf());
  const BboxNmsOpConf& conf = op_conf().bbox_nms_conf();
  EnrollInputBn("bbox", false);
  EnrollInputBn("bbox_score", false);
  EnrollOutputBn("out_bbox", false);
  EnrollOutputBn("out_bbox_score", false);
  if (conf.has_num_classes()) { EnrollOutputBn("out_bbox_label", false); }
}

const PbMessage& BboxNmsOp::GetCustomizedConf() const { return op_conf().bbox_nms_conf(); }

void BboxNmsOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                               const ParallelContext* parallel_ctx) const {
  const BboxNmsOpConf& conf = op_conf().bbox_nms_conf();
  // input: bbox_score (N, R, K) or (R, K) T
  // When the shape dims is (R, K), the image index info which it is belonged to is
  // hidden in the blob header field RecordId, also applies to bbox blob
  const BlobDesc* scores_blob_desc = GetBlobDesc4BnInOp("bbox_score");
  // input: bbox (N, R, 4 or K*4) or (R, 4 or K*4) T
  // Bbox may has only a pred for all classes or one pred for each class
  const BlobDesc* bbox_blob_desc = GetBlobDesc4BnInOp("bbox");
  CHECK_EQ(scores_blob_desc->data_type(), bbox_blob_desc->data_type());
  const int64_t num_axes = scores_blob_desc->shape().NumAxes();
  FOR_RANGE(int64_t, i, 0, num_axes - 1) {
    CHECK_EQ(scores_blob_desc->shape().At(i), bbox_blob_desc->shape().At(i));
  }
  const int64_t num_classes = scores_blob_desc->shape().At(num_axes - 1);
  if (conf.has_num_classes()) { CHECK_EQ(conf.num_classes(), num_classes); }
  CHECK(bbox_blob_desc->shape().At(num_axes - 1) == 4
        || bbox_blob_desc->shape().At(num_axes - 1) == 4 * num_classes);

  if (num_axes == 2) {
    // input has no image dim
    const int64_t num_images = Global<JobDesc>::Get()->DevicePieceSize4ParallelCtx(*parallel_ctx);
    const int64_t total_top_n = num_images * conf.post_nms_top_n();
    // output: out_bbox (R`, K*4) T
    BlobDesc* out_bbox_blob_desc = GetBlobDesc4BnInOp("out_bbox");
    out_bbox_blob_desc->mut_shape() = Shape({total_top_n, num_classes * 4});
    out_bbox_blob_desc->set_data_type(bbox_blob_desc->data_type());
    out_bbox_blob_desc->set_has_dim0_valid_num_field(true);
    out_bbox_blob_desc->set_has_record_id_in_device_piece_field(
        bbox_blob_desc->has_record_id_in_device_piece_field());
    // output: out_bbox_score (R`, K) T
    BlobDesc* out_bbox_score_blob_desc = GetBlobDesc4BnInOp("out_bbox_score");
    *out_bbox_score_blob_desc = *out_bbox_blob_desc;
    out_bbox_score_blob_desc->mut_shape() = Shape({total_top_n, num_classes});
    // output: out_bbox_label (R`,)
    BlobDesc* out_bbox_label_blob_desc = GetBlobDesc4BnInOp("out_bbox_label");
    if (out_bbox_label_blob_desc) {
      *out_bbox_label_blob_desc = *out_bbox_blob_desc;
      out_bbox_label_blob_desc->mut_shape() = Shape({total_top_n});
      out_bbox_label_blob_desc->set_data_type(DataType::kInt32);
    }
  } else if (num_axes == 3) {
    // input has image dim
    CHECK(!bbox_blob_desc->has_record_id_in_device_piece_field());
    CHECK(!scores_blob_desc->has_record_id_in_device_piece_field());
    const int64_t num_images = bbox_blob_desc->shape().At(0);
    const int64_t post_nms_n = conf.post_nms_top_n();
    // output: out_bbox (N, R`, K*4) T
    BlobDesc* out_bbox_blob_desc = GetBlobDesc4BnInOp("out_bbox");
    out_bbox_blob_desc->mut_shape() = Shape({num_images, post_nms_n, num_classes * 4});
    out_bbox_blob_desc->set_data_type(bbox_blob_desc->data_type());
    out_bbox_blob_desc->set_has_dim1_valid_num_field(true);
    // output: out_bbox_score (N, R`, K) T
    BlobDesc* out_bbox_score_blob_desc = GetBlobDesc4BnInOp("out_bbox_score");
    *out_bbox_score_blob_desc = *out_bbox_blob_desc;
    out_bbox_score_blob_desc->mut_shape() = Shape({num_images, post_nms_n, num_classes});
    // output: out_bbox_label (N, R`)
    BlobDesc* out_bbox_label_blob_desc = GetBlobDesc4BnInOp("out_bbox_label");
    if (out_bbox_label_blob_desc) {
      *out_bbox_label_blob_desc = *out_bbox_blob_desc;
      out_bbox_label_blob_desc->mut_shape() = Shape({num_images, post_nms_n});
      out_bbox_label_blob_desc->set_data_type(DataType::kInt32);
    }
  }
}

REGISTER_OP(OperatorConf::kBboxNmsConf, BboxNmsOp);

}  // namespace oneflow