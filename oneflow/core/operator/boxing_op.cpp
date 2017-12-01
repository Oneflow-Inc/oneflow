#include "oneflow/core/operator/boxing_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void BoxingOp::InitFromOpConf() {
  CHECK(op_conf().has_boxing_conf());
  auto boxing_conf = op_conf().boxing_conf();

  bool has_diff = JobDesc::Singleton()->is_train();
  for (int64_t i = 0; i < boxing_conf.in_num(); ++i) {
    EnrollInputBn("in_" + std::to_string(i), has_diff);
  }
  if (boxing_conf.in_box_case() == BoxingOpConf::kConcatBox
      && boxing_conf.out_box_case() == BoxingOpConf::kCloneBox) {
    EnrollDataTmpBn("middle");
  }
  for (int64_t i = 0; i < boxing_conf.out_num(); ++i) {
    EnrollOutputBn("out_" + std::to_string(i), has_diff);
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
