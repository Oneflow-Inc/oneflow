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
  CHECK_GT(conf.nms_threshold(), 0.f);
  if (conf.has_pre_nms_score_threshold()) { CHECK_GT(conf.pre_nms_score_threshold(), 0.f); }
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
  if (scores_blob_desc->has_record_id_in_device_piece_field()) {
    CHECK(bbox_blob_desc->has_record_id_in_device_piece_field());
  }
  int64_t keep_num_per_image = 0;
  if (conf.has_image_top_n()) {
    CHECK_GT(conf.image_top_n(), 0);
    keep_num_per_image = conf.image_top_n();
  } else {
    if (conf.has_post_nms_top_n()) {
      CHECK_GT(conf.post_nms_top_n(), 0);
      keep_num_per_image = num_classes * conf.post_nms_top_n();
      if (conf.has_pre_nms_top_n()) { CHECK_GE(conf.pre_nms_top_n(), conf.post_nms_top_n()); }
    } else {
      if (conf.has_pre_nms_top_n()) {
        CHECK_GT(conf.pre_nms_top_n(), 0);
        keep_num_per_image = num_classes * conf.pre_nms_top_n();
      }
    }
  }
  CHECK_GT(keep_num_per_image, 0);
  if (num_axes == 2) {
    // input has no image dim
    const int64_t num_images = Global<JobDesc>::Get()->DevicePieceSize4ParallelCtx(*parallel_ctx);
    const int64_t total_top_n = keep_num_per_image * num_images;
    // output: out_bbox (R`, 4) T
    BlobDesc* out_bbox_blob_desc = GetBlobDesc4BnInOp("out_bbox");
    out_bbox_blob_desc->mut_shape() = Shape({total_top_n, 4});
    out_bbox_blob_desc->set_data_type(bbox_blob_desc->data_type());
    out_bbox_blob_desc->mut_dim0_inner_shape() = Shape({1, total_top_n});
    out_bbox_blob_desc->set_has_dim0_valid_num_field(true);
    out_bbox_blob_desc->set_has_record_id_in_device_piece_field(
        bbox_blob_desc->has_record_id_in_device_piece_field());
    // output: out_bbox_score (R`,) T
    BlobDesc* out_bbox_score_blob_desc = GetBlobDesc4BnInOp("out_bbox_score");
    *out_bbox_score_blob_desc = *out_bbox_blob_desc;
    out_bbox_score_blob_desc->mut_shape() = Shape({total_top_n});
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
    // output: out_bbox (N, R`, 4) T
    BlobDesc* out_bbox_blob_desc = GetBlobDesc4BnInOp("out_bbox");
    out_bbox_blob_desc->mut_shape() = Shape({num_images, keep_num_per_image, 4});
    out_bbox_blob_desc->set_data_type(bbox_blob_desc->data_type());
    out_bbox_blob_desc->set_has_dim1_valid_num_field(true);
    // output: out_bbox_score (N, R`) T
    BlobDesc* out_bbox_score_blob_desc = GetBlobDesc4BnInOp("out_bbox_score");
    *out_bbox_score_blob_desc = *out_bbox_blob_desc;
    out_bbox_score_blob_desc->mut_shape() = Shape({num_images, keep_num_per_image});
    // output: out_bbox_label (N, R`)
    BlobDesc* out_bbox_label_blob_desc = GetBlobDesc4BnInOp("out_bbox_label");
    if (out_bbox_label_blob_desc) {
      *out_bbox_label_blob_desc = *out_bbox_blob_desc;
      out_bbox_label_blob_desc->mut_shape() = Shape({num_images, keep_num_per_image});
      out_bbox_label_blob_desc->set_data_type(DataType::kInt32);
    }
  } else {
    UNIMPLEMENTED();
  }
}

void BboxNmsOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const BlobDesc* bbox_blob_desc = GetBlobDesc4BnInOp("bbox");
  const BlobDesc* scores_blob_desc = GetBlobDesc4BnInOp("bbox_score");
  const int64_t num_axes = scores_blob_desc->shape().NumAxes();
  BboxNmsKernelConf* bbox_nms_conf = kernel_conf->mutable_bbox_nms_conf();
  bbox_nms_conf->set_num_classes(scores_blob_desc->shape().At(num_axes - 1));
  bbox_nms_conf->set_need_broadcast(bbox_blob_desc->shape().At(num_axes - 1) == 4);
  if (num_axes == 2) {
    bbox_nms_conf->set_device_piece_size(
        Global<JobDesc>::Get()->DevicePieceSize4ParallelCtx(*parallel_ctx));
  }
}

REGISTER_OP(OperatorConf::kBboxNmsConf, BboxNmsOp);

}  // namespace oneflow