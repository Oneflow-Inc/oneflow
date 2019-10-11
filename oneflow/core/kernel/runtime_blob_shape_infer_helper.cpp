#include "oneflow/core/kernel/runtime_blob_shape_infer_helper.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

RuntimeBlobShapeInferHelper::RuntimeBlobShapeInferHelper(const OperatorConf& op_conf,
                                                         const JobDesc* job_desc) {
  op_ = ConstructOp(op_conf, job_desc);
  for (const auto& ibn : op_->input_bns()) { ibns_.insert(ibn); }
  auto* map = sbp_signature_.mutable_bn_in_op2sbp_parallel();
  op_->ForEachBnInOp([&](const std::string& bn_in_op) {
    bn_in_op2blob_desc_[bn_in_op].reset();
    (*map)[bn_in_op].mutable_split_parallel()->set_axis(0);
  });
  parallel_ctx_.set_parallel_id(0);
  parallel_ctx_.set_parallel_num(1);
}

void RuntimeBlobShapeInferHelper::InferDenseShape(
    std::function<Blob*(const std::string&)> BnInOp2Blob) {
  HashSet<const BlobDesc*> updated_input_blobs;
  auto BlobDesc4BnInOp = [&](const std::string& bn_in_op) -> BlobDesc* {
    BlobDesc* blob_desc = bn_in_op2blob_desc_.at(bn_in_op).get();
    const Blob* blob = BnInOp2Blob(bn_in_op);
    if (blob == nullptr) { return nullptr; }
    if (blob_desc == nullptr) {
      const RtBlobDesc& rt_blob_desc = blob->blob_desc();
      blob_desc = new BlobDesc(rt_blob_desc.body(), rt_blob_desc.num_of_lod_levels(),
                               rt_blob_desc.is_body_disabled(), rt_blob_desc.is_dynamic());
      bn_in_op2blob_desc_.at(bn_in_op).reset(blob_desc);
    }
    if (updated_input_blobs.find(blob_desc) == updated_input_blobs.end()
        && ibns_.find(bn_in_op) != ibns_.end()) {
      if (blob_desc->num_of_lod_levels() > 0) {
        CHECK_EQ(blob_desc->shape().NumAxes(),
                 blob->shape().NumAxes() + (blob_desc->num_of_lod_levels() - 1));
      } else {
        CHECK_EQ(blob_desc->shape().NumAxes(), blob->shape().NumAxes());
      }
      blob_desc->mut_shape() =
          CreateLeftExtendedShape(blob->shape(), blob->static_shape().NumAxes());
      updated_input_blobs.insert(blob_desc);
    }
    return blob_desc;
  };
  CHECK_JUST(op_->InferOutBlobDescsIf(BlobDesc4BnInOp, &parallel_ctx_, &sbp_signature_,
                                      [](OpContext*) {}));
  for (const auto& obn : op_->output_bns()) {
    auto* blob = BnInOp2Blob(obn);
    if (blob == nullptr) { continue; }
    const auto& blob_desc = bn_in_op2blob_desc_.at(obn);
    CHECK_EQ(blob->data_type(), blob_desc->data_type());
    CHECK_EQ(blob->blob_desc().is_dynamic(), blob_desc->is_dynamic());
    CHECK_EQ(blob->blob_desc().is_body_disabled(), blob_desc->is_body_disabled());
    if (blob->blob_desc().is_dynamic()) {
      Shape shape(blob_desc->shape());
      int64_t num_of_lod_levels = blob->blob_desc().num_of_lod_levels();
      if (num_of_lod_levels > 0) {
        FOR_RANGE(int, i, 0, num_of_lod_levels - 1) { CHECK_EQ(shape.At(i), 1); }
        shape = Shape({shape.dim_vec().begin() + num_of_lod_levels - 1, shape.dim_vec().end()});
      }
      blob->dense_shape_mut_view().set_shape(shape);
    } else {
      CHECK_EQ(blob->shape(), blob_desc->shape());
    }
  }
}

}  // namespace oneflow
