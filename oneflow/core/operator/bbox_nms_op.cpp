#include "oneflow/core/operator/bbox_nms_op.h"

namespace oneflow {

void BboxNmsOp::InitFromOpConf() {
  CHECK(op_conf().has_bbox_nms_conf());
  EnrollInputBn("bbox", false);
  EnrollInputBn("bbox_score", false);
  EnrollOutputBn("out_bbox", false);
  EnrollOutputBn("bbox_score", false);
  EnrollOutputBn("class_bbox_cnt", false);
}

const PbMessage& BboxNmsOp::GetCustomizedConf() const { return op_conf().bbox_nms_conf(); }

void BboxNmsOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                               const ParallelContext* parallel_ctx) const {
  const BlobDesc* bbox_blob_desc = GetBlobDesc4BnInOp("bbox");
  const BlobDesc* scores_blob_desc = GetBlobDesc4BnInOp("bbox_score");
  const int64_t images_num = scores_blob_desc->shape().At(0);
  const int64_t boxes_num = scores_blob_desc->shape().At(1);
  const int64_t class_num = scores_blob_desc->shape().At(2);
  CHECK_EQ(bbox_blob_desc->shape().At(0), scores_blob_desc->shape().At(0));
  CHECK_EQ(bbox_blob_desc->shape().At(1), scores_blob_desc->shape().At(1));
  CHECK(bbox_blob_desc->shape().At(2) == 4 || bbox_blob_desc->shape().At(2) == 4 * class_num);

  // output: bbox_score (n,c,r)
  BlobDesc* bbox_score_blob_desc = GetBlobDesc4BnInOp("bbox_score");
  bbox_score_blob_desc->mut_shape() = Shape({images_num, class_num, boxes_num});
  bbox_score_blob_desc->set_data_type(scores_blob_desc->data_type());

  // output: out_bbox (n,c,r*4)
  BlobDesc* out_bbox_blob_desc = GetBlobDesc4BnInOp("out_bbox");
  out_bbox_blob_desc->mut_shape() = Shape({images_num, class_num, boxes_num * 4});
  out_bbox_blob_desc->set_data_type(bbox_blob_desc->data_type());

  // output: class_bbox_cnt (n,c)
  BlobDesc* class_bbox_cnt_blob_desc = GetBlobDesc4BnInOp("class_bbox_cnt");
  class_bbox_cnt_blob_desc->mut_shape() = Shape({images_num, class_num});
  class_bbox_cnt_blob_desc->set_data_type(DataType::kInt32);

  // datatmp: pre_nms_index_slice (c, r)
  BlobDesc* pre_nms_index_blob_desc = GetBlobDesc4BnInOp("pre_nms_index_slice");
  pre_nms_index_blob_desc->mut_shape() = Shape({class_num, boxes_num});
  pre_nms_index_blob_desc->set_data_type(DataType::kInt32);
  // datatmp: post_nms_index_slice (c, r)
  *GetBlobDesc4BnInOp("post_nms_index_slice") = *pre_nms_index_blob_desc;

  // output: cls_score_index_slice (c)
  BlobDesc* cls_score_index_slice_blob_desc = GetBlobDesc4BnInOp("cls_score_index_slice");
  cls_score_index_slice_blob_desc->mut_shape() = Shape({class_num});
  cls_score_index_slice_blob_desc->set_data_type(DataType::kInt32);
}

REGISTER_OP(OperatorConf::kBboxNmsConf, BboxNmsOp);

}  // namespace oneflow