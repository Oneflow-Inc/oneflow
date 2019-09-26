#include "oneflow/core/kernel/runtime_blob_shape_infer_helper.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

RuntimeBlobShapeInferHelper::RuntimeBlobShapeInferHelper(const OperatorConf& op_conf,
                                                         const JobDesc* job_desc) {
  op_ = ConstructOp(op_conf, job_desc);
  for (const auto& ibn : op_->input_bns()) { ibns_.insert(ibn); }
  for (const auto& ibn : op_->input_bns()) { bn_in_op2blob_desc_[ibn].reset(); }
  for (const auto& obn : op_->output_bns()) { bn_in_op2blob_desc_[obn].reset(); }
  parallel_ctx_.set_parallel_id(0);
  parallel_ctx_.set_parallel_num(1);
  for (const auto& ibn : op_->input_bns()) {
    (*sbp_signature_.mutable_bn_in_op2sbp_parallel())[ibn].mutable_split_parallel()->set_axis(0);
  }
  for (const auto& obn : op_->output_bns()) {
    (*sbp_signature_.mutable_bn_in_op2sbp_parallel())[obn].mutable_split_parallel()->set_axis(0);
  }
}

void RuntimeBlobShapeInferHelper::InferDenseShape(
    std::function<Blob*(const std::string&)> BnInOp2Blob) {
  HashSet<const BlobDesc*> updated_input_blobs;
  auto BlobDesc4BnInOp = [&](const std::string& bn_in_op) {
    BlobDesc* blob_desc = bn_in_op2blob_desc_.at(bn_in_op).get();
    if (blob_desc == nullptr) {
      const RtBlobDesc& rt_blob_desc = BnInOp2Blob(bn_in_op)->blob_desc();
      blob_desc = new BlobDesc(rt_blob_desc.body(), rt_blob_desc.num_of_lod_levels(),
                               rt_blob_desc.is_body_disabled(), rt_blob_desc.is_dynamic());
      bn_in_op2blob_desc_.at(bn_in_op).reset(blob_desc);
    }
    if (updated_input_blobs.find(blob_desc) == updated_input_blobs.end()
        && ibns_.find(bn_in_op) != ibns_.end()) {
      const Blob* blob = BnInOp2Blob(bn_in_op);
      CHECK_EQ(blob_desc->shape().NumAxes(), blob->shape().NumAxes());
      blob_desc->mut_shape() = blob->shape();
      updated_input_blobs.insert(blob_desc);
    }
    return blob_desc;
  };
  CHECK_JUST(op_->InferOutBlobDescsIf(BlobDesc4BnInOp, &parallel_ctx_, &sbp_signature_,
                                      [](OpContext*) {}));
  for (const auto& obn : op_->output_bns()) {
    auto* blob = BnInOp2Blob(obn);
    const auto& blob_desc = bn_in_op2blob_desc_.at(obn);
    CHECK_EQ(blob->data_type(), blob_desc->data_type());
    CHECK_EQ(blob->blob_desc().is_dynamic(), blob_desc->is_dynamic());
    CHECK_EQ(blob->blob_desc().is_body_disabled(), blob_desc->is_body_disabled());
    if (blob->blob_desc().is_dynamic()) {
      blob->dense_shape_mut_view().set_shape(blob_desc->shape());
    } else {
      CHECK_EQ(blob->shape(), blob_desc->shape());
    }
  }
}

}  // namespace oneflow
