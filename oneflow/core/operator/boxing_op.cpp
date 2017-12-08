#include "oneflow/core/operator/boxing_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void BoxingOp::InitFromOpConf() {
  CHECK(op_conf().has_boxing_conf());
  const BoxingOpConf& boxing_conf = op_conf().boxing_conf();

  for (int64_t i = 0; i < boxing_conf.in_num(); ++i) {
    EnrollInputBn("in_" + std::to_string(i), false);
  }
  if (boxing_conf.in_box_case() == BoxingOpConf::kAddBox
      && boxing_conf.out_box_case() == BoxingOpConf::kSplitBox) {
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

void BoxingOp::GenBoxingInfo(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const std::vector<std::string>& bns, int32_t axis, bool is_concat_or_split,
    BoxingInfo* boxing_info) const {
  const BlobDesc* first_blob = GetBlobDesc4BnInOp(bns.front());
  int32_t total_seg_num = 1;
  int64_t seg_size_acc = 0;
  if (is_concat_or_split) {
    if (axis != 0) { total_seg_num = first_blob->shape().Count(0, axis); }
    for (const std::string& bn : bns) {
      const BlobDesc* blob = GetBlobDesc4BnInOp(bn);
      int64_t seg_subsize = blob->shape().Count(axis);
      boxing_info->add_size_of_subseg(seg_subsize);
      boxing_info->add_offset_of_subseg(seg_size_acc);
      seg_size_acc += seg_subsize;
    }
  } else {
    seg_size_acc = first_blob->shape().Count(0);
    boxing_info->add_size_of_subseg(seg_size_acc);
    boxing_info->add_offset_of_subseg(0);
  }
  boxing_info->set_total_seg_num(total_seg_num);
  boxing_info->set_size_of_per_seg(seg_size_acc);
}

void BoxingOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const BoxingOpConf& conf = op_conf().boxing_conf();
  BoxingInfo* in_info = kernel_conf->mutable_boxing_conf()->mutable_in_info();
  int32_t concat_axis = 0;
  bool is_concat = (conf.in_box_case() == BoxingOpConf::kConcatBox);
  if (is_concat) { concat_axis = conf.concat_box().axis(); }
  GenBoxingInfo(GetBlobDesc4BnInOp, input_bns(), concat_axis, is_concat,
                in_info);

  BoxingInfo* out_info = kernel_conf->mutable_boxing_conf()->mutable_out_info();
  int32_t split_axis = 0;
  bool is_split = (conf.out_box_case() == BoxingOpConf::kSplitBox);
  if (is_split) { split_axis = conf.split_box().axis(); }
  GenBoxingInfo(GetBlobDesc4BnInOp, output_bns(), split_axis, is_split,
                out_info);
}

void BoxingOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BoxingOpConf& conf = op_conf().boxing_conf();

  std::vector<int64_t> data_tmp_blob_shape_vec =
      GetBlobDesc4BnInOp(input_bns().front())->shape().dim_vec();
  int32_t concat_axis = 0;
  if (conf.in_box_case() == BoxingOpConf::kConcatBox) {
    concat_axis = conf.concat_box().axis();
    CHECK_GE(concat_axis, 0);
    FOR_RANGE(size_t, ib_idx, 1, input_bns().size()) {
      const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp(input_bns().at(ib_idx));
      const std::vector<int64_t>& in_blob_shape_vec =
          in_blob_desc->shape().dim_vec();
      CHECK_LT(concat_axis, in_blob_shape_vec.size());
      FOR_RANGE(size_t, i, 0, in_blob_shape_vec.size()) {
        if (i == concat_axis) {
          data_tmp_blob_shape_vec[i] += in_blob_shape_vec[i];
        } else {
          CHECK_EQ(data_tmp_blob_shape_vec[i], in_blob_shape_vec[i]);
        }
      }
    }
  }

  bool has_data_id = GetBlobDesc4BnInOp(input_bns().front())->has_data_id();
  BlobDesc* first_in_blob = GetBlobDesc4BnInOp(input_bns().front());
  CHECK_NE(conf.out_box_case(), BoxingOpConf::OUT_BOX_NOT_SET);
  if (conf.in_box_case() == BoxingOpConf::kAddBox
      && conf.out_box_case() == BoxingOpConf::kSplitBox) {
    BlobDesc* data_tmp_blob_desc = GetBlobDesc4BnInOp(SoleDtbn());
    data_tmp_blob_desc->set_has_data_id(false);
    data_tmp_blob_desc->set_data_type(first_in_blob->data_type());
    data_tmp_blob_desc->mut_shape() = Shape(data_tmp_blob_shape_vec);
  }

  if (conf.out_box_case() == BoxingOpConf::kSplitBox) {
    const BoxSplitConf& split_conf = conf.split_box();
    std::vector<int64_t> output_shape_vec = data_tmp_blob_shape_vec;
    CHECK_GE(split_conf.axis(), 0);
    CHECK_LT(split_conf.axis(), output_shape_vec.size());
    FOR_RANGE(size_t, i, 0, output_bns().size()) {
      BlobDesc* out_blob_desc = GetBlobDesc4BnInOp(output_bns().at(i));
      out_blob_desc->set_has_data_id(has_data_id);
      out_blob_desc->set_data_type(first_in_blob->data_type());
      output_shape_vec[split_conf.axis()] = split_conf.part_num(i);
      out_blob_desc->mut_shape() = Shape(output_shape_vec);
    }
  } else if (conf.out_box_case() == BoxingOpConf::kCloneBox) {
    for (const std::string& obn : output_bns()) {
      BlobDesc* out_blob_desc = GetBlobDesc4BnInOp(obn);
      out_blob_desc->set_has_data_id(has_data_id);
      out_blob_desc->set_data_type(first_in_blob->data_type());
      out_blob_desc->mut_shape() = Shape(data_tmp_blob_shape_vec);
    }
  } else {
    UNEXPECTED_RUN();
  }
}

REGISTER_OP(OperatorConf::kBoxingConf, BoxingOp);

}  // namespace oneflow
