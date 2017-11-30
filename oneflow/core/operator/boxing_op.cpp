#include "oneflow/core/operator/boxing_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void BoxingOp::InitFromOpConf() {
  CHECK(op_conf().has_boxing_conf());
  auto boxing_conf = op_conf().boxing_conf();

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

void BoxingOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  DataType dtype = GetBlobDesc4BnInOp(input_bns().front())->data_type();
  kernel_conf->mutable_boxing_conf()->set_data_type(dtype);
  const BlobDesc* first_in_blob = GetBlobDesc4BnInOp(input_bns().front());
  int64_t in_seg_cnt = 1;
  int64_t in_seg_size = 0;
  std::vector<int64_t> in_offset_in_seg;
  std::vector<int64_t> in_size_in_seg;
  auto conf = op_conf().boxing_conf();
  if (conf.in_box_case() == BoxingOpConf::kConcatBox) {
    int32_t concat_axis = conf.concat_box().axis();
    if (concat_axis != 0) {
      in_seg_cnt = first_in_blob->shape().Count(0, concat_axis);
    }
    for (const std::string& ibn : input_bns()) {
      const BlobDesc* in_blob = GetBlobDesc4BnInOp(ibn);
      int64_t in_seg_offset = in_blob->shape().Count(concat_axis);
      in_size_in_seg.push_back(in_seg_offset);
      in_offset_in_seg.push_back(in_seg_size);
      in_seg_size += in_seg_offset;
    }
  } else {
    in_seg_size = first_in_blob->shape().Count(0);
    if (first_in_blob->shape().NumAxes() > 1) {
      in_offset_in_seg = {in_seg_size};
    }
  }
  kernel_conf->mutable_boxing_conf()->set_in_seg_cnt(in_seg_cnt);
  kernel_conf->mutable_boxing_conf()->set_in_seg_size(in_seg_size);
  *(kernel_conf->mutable_boxing_conf()->mutable_in_offset_in_seg()) =
      StdVec2PbRf<int64_t>(in_offset_in_seg);
  *(kernel_conf->mutable_boxing_conf()->mutable_in_size_in_seg()) =
      StdVec2PbRf<int64_t>(in_size_in_seg);

  const BlobDesc* first_out_blob = GetBlobDesc4BnInOp(output_bns().front());
  std::vector<int64_t> out_size_in_seg;
  int64_t out_seg_cnt = 1;
  int64_t out_seg_size = 0;
  if (conf.out_box_case() == BoxingOpConf::kSplitBox) {
    int32_t split_axis = conf.split_box().axis();
    if (split_axis != 0) {
      out_seg_cnt = first_out_blob->shape().Count(0, split_axis);
    }
    for (const std::string& obn : output_bns()) {
      const BlobDesc* out_blob = GetBlobDesc4BnInOp(obn);
      int64_t out_seg_offset = out_blob->shape().Count(split_axis);
      out_size_in_seg.push_back(out_seg_offset);
      out_seg_size += out_seg_offset;
    }
  } else {
    out_seg_size = first_out_blob->shape().Count(0);
    out_size_in_seg = {out_seg_size};
  }
  kernel_conf->mutable_boxing_conf()->set_out_seg_cnt(out_seg_cnt);
  kernel_conf->mutable_boxing_conf()->set_out_seg_size(out_seg_size);
  *(kernel_conf->mutable_boxing_conf()->mutable_out_size_in_seg()) =
      StdVec2PbRf<int64_t>(out_size_in_seg);
}

void BoxingOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) {
  auto conf = op_conf().boxing_conf();

  auto in_box_case = conf.in_box_case();
  std::vector<int64_t> data_tmp_blob_shape_vec =
      GetBlobDesc4BnInOp(input_bns().at(0))->shape().dim_vec();
  int32_t concat_axis = 0;
  if (in_box_case == BoxingOpConf::kConcatBox) {
    concat_axis = conf.concat_box().axis();
    CHECK_GE(concat_axis, 0);
  }
  // check datatype of input desc && calculate the shape of data_tmp
  for (size_t ib_idx = 1; ib_idx < input_bns().size(); ++ib_idx) {
    const BlobDesc* ib_desc = GetBlobDesc4BnInOp(input_bns().at(ib_idx));
    const std::vector<int64_t>& ib_shape_vec = ib_desc->shape().dim_vec();
    CHECK_LT(concat_axis, ib_shape_vec.size());
    // if it is a concat-box, accumulate the dimensions on concat-axis.
    // otherwise only check all boxes are in the same shape.
    for (size_t i = 0; i < ib_shape_vec.size(); ++i) {
      if (in_box_case == BoxingOpConf::kConcatBox && i == concat_axis) {
        data_tmp_blob_shape_vec[i] += ib_shape_vec[i];
      } else {
        CHECK_EQ(data_tmp_blob_shape_vec[i], ib_shape_vec[i]);
      }
    }
  }

  auto out_box_case = conf.out_box_case();
  const bool has_data_id = GetBlobDesc4BnInOp(input_bns().at(0))->has_data_id();
  CHECK_NE(out_box_case, BoxingOpConf::OUT_BOX_NOT_SET);
  if (in_box_case == BoxingOpConf::kConcatBox
      && out_box_case == BoxingOpConf::kCloneBox) {
    BlobDesc* dtb_desc = GetBlobDesc4BnInOp(SoleDtbn());
    dtb_desc->set_has_data_id(has_data_id);
    dtb_desc->mut_shape() = Shape(data_tmp_blob_shape_vec);
  }

  // infer desc of out blobs
  if (out_box_case == BoxingOpConf::kSplitBox) {
    auto split_conf = conf.split_box();
    std::vector<int64_t> output_shape_vec = data_tmp_blob_shape_vec;
    CHECK_GE(split_conf.axis(), 0);
    CHECK_LT(split_conf.axis(), output_shape_vec.size());
    CHECK_EQ(split_conf.part_num_size(), output_shape_vec.size());
    for (size_t i = 0; i < output_bns().size(); ++i) {
      BlobDesc* ob_desc = GetBlobDesc4BnInOp(output_bns().at(i));
      ob_desc->set_has_data_id(has_data_id);
      output_shape_vec[split_conf.axis()] = split_conf.part_num(i);
      ob_desc->mut_shape() = Shape(output_shape_vec);
    }
  } else if (out_box_case == BoxingOpConf::kCloneBox) {
    for (const std::string& obn : output_bns()) {
      GetBlobDesc4BnInOp(obn)->mut_shape() = Shape(data_tmp_blob_shape_vec);
    }
  } else {
    UNEXPECTED_RUN();
  }
}

REGISTER_OP(OperatorConf::kBoxingConf, BoxingOp);

}  // namespace oneflow
