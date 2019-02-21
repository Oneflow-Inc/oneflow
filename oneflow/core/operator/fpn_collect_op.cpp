#include "oneflow/core/operator/fpn_collect_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void FpnCollectOp::InitFromOpConf() {
  CHECK(op_conf().has_fpn_collect_conf());
  EnrollRepeatedInputBn("rpn_rois_fpn");
  EnrollRepeatedInputBn("rpn_roi_probs_fpn");
  EnrollOutputBn("out");
  EnrollDataTmpBn("roi_inds");
}

const PbMessage& FpnCollectOp::GetCustomizedConf() const { return op_conf().fpn_collect_conf(); }

void FpnCollectOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext* parallel_ctx) const {
  CHECK_EQ(parallel_ctx->policy(), kDataParallel);
  const FpnCollectOpConf& conf = op_conf().fpn_collect_conf();
  const int64_t num_layers = conf.rpn_rois_fpn_size();
  CHECK_EQ(conf.rpn_roi_probs_fpn_size(), num_layers);
  BalancedSplitter bs(Global<JobDesc>::Get()->RecordPieceSize(), parallel_ctx->parallel_num());
  int64_t total_top_n = conf.top_n_per_image() * bs.At(parallel_ctx->parallel_id()).size();
  int64_t max_num_rois_per_layer = 0;
  int64_t total_num_rois = 0;
  DataType data_type = DataType::kInvalidDataType;
  bool has_record_id_in_device_piece_field = false;
  bool set_has_record_id_in_device_piece_field = false;
  FOR_RANGE(size_t, i, 0, num_layers) {
    // input: rpn_rois_fpn_i (R, 5) T
    BlobDesc* rois_blob_desc_i = GetBlobDesc4BnInOp(GenRepeatedBn("rpn_rois_fpn", i));
    // input: rpn_rois_fpn_i (R) T
    BlobDesc* roi_probs_blob_desc_i = GetBlobDesc4BnInOp(GenRepeatedBn("rpn_roi_probs_fpn", i));
    CHECK_EQ(rois_blob_desc_i->shape().At(0), roi_probs_blob_desc_i->shape().At(0));
    CHECK_EQ(rois_blob_desc_i->data_type(), roi_probs_blob_desc_i->data_type());
    CHECK_EQ(rois_blob_desc_i->has_dim0_valid_num_field(),
             roi_probs_blob_desc_i->has_dim0_valid_num_field());
    CHECK_EQ(rois_blob_desc_i->has_record_id_in_device_piece_field(),
             roi_probs_blob_desc_i->has_record_id_in_device_piece_field());
    max_num_rois_per_layer = std::max(max_num_rois_per_layer, rois_blob_desc_i->shape().At(0));
    total_num_rois += rois_blob_desc_i->shape().At(0);
    if (data_type == DataType::kInvalidDataType) {
      data_type = rois_blob_desc_i->data_type();
    } else {
      CHECK_EQ(data_type, rois_blob_desc_i->data_type());
    }
    if (!set_has_record_id_in_device_piece_field) {
      has_record_id_in_device_piece_field = rois_blob_desc_i->has_record_id_in_device_piece_field();
      set_has_record_id_in_device_piece_field = true;
    } else {
      CHECK_EQ(has_record_id_in_device_piece_field,
               rois_blob_desc_i->has_record_id_in_device_piece_field());
    }
  }
  total_top_n = std::min(total_top_n, total_num_rois);

  // output: out (total_top_n, 5) T
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  out_blob_desc->mut_shape() = Shape({total_top_n, 5});
  out_blob_desc->set_data_type(data_type);
  out_blob_desc->mut_dim0_inner_shape() = Shape({1, total_top_n});
  out_blob_desc->set_has_dim0_valid_num_field(true);
  out_blob_desc->set_has_record_id_in_device_piece_field(has_record_id_in_device_piece_field);

  // datatmp: roi_inds (num_layers, max_num_rois_per_layer) int32
  BlobDesc* roi_inds_blob_desc = GetBlobDesc4BnInOp("roi_inds");
  roi_inds_blob_desc->mut_shape() = Shape({num_layers, max_num_rois_per_layer});
  roi_inds_blob_desc->set_data_type(DataType::kInt32);
}

REGISTER_CPU_OP(OperatorConf::kFpnCollectConf, FpnCollectOp);

}  // namespace oneflow
