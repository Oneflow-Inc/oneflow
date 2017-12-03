#include "oneflow/core/operator/boxing_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/kernel/boxing_info.pb.h"

namespace oneflow {

void BoxingOp::InitFromOpConf() {
  CHECK(op_conf().has_boxing_conf());
  const BoxingOpConf& boxing_conf = op_conf().boxing_conf();

  for (int64_t i = 0; i < boxing_conf.in_num(); ++i) {
    EnrollInputBn("in_" + std::to_string(i), false);
  }
  if (boxing_conf.in_box_case() == BoxingOpConf::kConcatBox
      && boxing_conf.out_box_case() == BoxingOpConf::kCloneBox) {
    EnrollDataTmpBn("middle");
  }
  for (int64_t i = 0; i < boxing_conf.out_num(); ++i) {
    EnrollOutputBn("out_" + std::to_string(i), false);
  }
}

const PbMessage& BoxingOp::GetSpecialConf() const {
  return op_conf().boxing_conf();
}

std::string BoxingOp::ibn2lbn(const std::string& input_bn) const {
  return GetStringFromSpecialConf("lbn");
}

std::string BoxingOp::obn2lbn(const std::string& output_bn) const {
  return GetStringFromSpecialConf("lbn");
}

void BoxingOp::GetBoxingInfo(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const std::vector<std::string>& bns, BoxingInfo* boxing_info, int32_t axis,
    bool is_concat_or_split) const {
  const BlobDesc* first_blob = GetBlobDesc4BnInOp(bns.front());
  int32_t seg_num = 1;
  int64_t seg_size_acc = 0;
  if (is_concat_or_split) {
    if (axis != 0) { seg_num = first_blob->shape().Count(0, axis); }
    for (const std::string& bn : bns) {
      const BlobDesc* blob = GetBlobDesc4BnInOp(bn);
      int64_t seg_subsize = blob->shape().Count(axis);
      boxing_info->add_size_in_seg(seg_subsize);
      boxing_info->add_offset_in_seg(seg_size_acc);
      seg_size_acc += seg_subsize;
    }
  } else {
    seg_size_acc = first_blob->shape().Count(0);
    boxing_info->add_size_in_seg(seg_size_acc);
    boxing_info->add_offset_in_seg(0);
  }
  boxing_info->set_seg_num(seg_num);
  boxing_info->set_seg_size(seg_size_acc);
}

void BoxingOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const BoxingOpConf& conf = op_conf().boxing_conf();
  BoxingInfo* in_info = kernel_conf->mutable_boxing_conf()->mutable_in_info();
  int32_t concat_axis = 0;
  bool is_concat_or_split = (conf.in_box_case() == BoxingOpConf::kConcatBox);
  if (is_concat_or_split) { concat_axis = conf.concat_box().axis(); }
  GetBoxingInfo(GetBlobDesc4BnInOp, input_bns(), in_info, concat_axis,
                is_concat_or_split);

  BoxingInfo* out_info = kernel_conf->mutable_boxing_conf()->mutable_out_info();
  int32_t split_axis = 0;
  is_concat_or_split = (conf.out_box_case() == BoxingOpConf::kSplitBox);
  if (is_concat_or_split) { split_axis = conf.split_box().axis(); }
  GetBoxingInfo(GetBlobDesc4BnInOp, output_bns(), out_info, split_axis,
                is_concat_or_split);
}

void BoxingOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) {
  const BoxingOpConf& conf = op_conf().boxing_conf();

  std::vector<int64_t> data_tmp_blob_shape_vec =
      GetBlobDesc4BnInOp(input_bns().at(0))->shape().dim_vec();
  int32_t concat_axis = 0;
  if (conf.in_box_case() == BoxingOpConf::kConcatBox) {
    concat_axis = conf.concat_box().axis();
    CHECK_GE(concat_axis, 0);
  }
  // check datatype of input desc && calculate the shape of data_tmp
  FOR_RANGE(size_t, ib_idx, 1, input_bns().size()) {
    const BlobDesc* ib_desc = GetBlobDesc4BnInOp(input_bns().at(ib_idx));
    const std::vector<int64_t>& ib_shape_vec = ib_desc->shape().dim_vec();
    CHECK_LT(concat_axis, ib_shape_vec.size());
    // if it is a concat-box, accumulate the dimensions on concat-axis.
    // otherwise only check all boxes are in the same shape.
    FOR_RANGE(size_t, i, 0, ib_shape_vec.size()) {
      if (conf.in_box_case() == BoxingOpConf::kConcatBox && i == concat_axis) {
        data_tmp_blob_shape_vec[i] += ib_shape_vec[i];
      } else {
        CHECK_EQ(data_tmp_blob_shape_vec[i], ib_shape_vec[i]);
      }
    }
  }

  bool has_data_id = GetBlobDesc4BnInOp(input_bns().at(0))->has_data_id();
  CHECK_NE(conf.out_box_case(), BoxingOpConf::OUT_BOX_NOT_SET);
  if (conf.in_box_case() == BoxingOpConf::kConcatBox
      && conf.out_box_case() == BoxingOpConf::kCloneBox) {
    BlobDesc* dtb_desc = GetBlobDesc4BnInOp(SoleDtbn());
    dtb_desc->set_has_data_id(has_data_id);
    dtb_desc->mut_shape() = Shape(data_tmp_blob_shape_vec);
  }

  // infer desc of out blobs
  if (conf.out_box_case() == BoxingOpConf::kSplitBox) {
    const BoxSplitConf& split_conf = conf.split_box();
    std::vector<int64_t> output_shape_vec = data_tmp_blob_shape_vec;
    CHECK_GE(split_conf.axis(), 0);
    CHECK_LT(split_conf.axis(), output_shape_vec.size());
    FOR_RANGE(size_t, i, 0, output_bns().size()) {
      BlobDesc* ob_desc = GetBlobDesc4BnInOp(output_bns().at(i));
      ob_desc->set_has_data_id(has_data_id);
      output_shape_vec[split_conf.axis()] = split_conf.part_num(i);
      ob_desc->mut_shape() = Shape(output_shape_vec);
    }
  } else if (conf.out_box_case() == BoxingOpConf::kCloneBox) {
    for (const std::string& obn : output_bns()) {
      GetBlobDesc4BnInOp(obn)->mut_shape() = Shape(data_tmp_blob_shape_vec);
    }
  } else {
    UNEXPECTED_RUN();
  }
}

REGISTER_OP(OperatorConf::kBoxingConf, BoxingOp);

}  // namespace oneflow
