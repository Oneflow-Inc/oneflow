#include "oneflow/core/operator/boxing_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void BoxingOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_boxing_conf());
  mut_op_conf() = op_conf;

  for (int64_t i = 0; i < op_conf.boxing_conf().in_num(); ++i) {
    EnrollInputBn("in_" + std::to_string(i));
  }
  if (op_conf.boxing_conf().in_box_case() == BoxingOpConf::kConcatBox
      && op_conf.boxing_conf().out_box_case() == BoxingOpConf::kCloneBox) {
    EnrollDataTmpBn("middle");
  }
  for (int64_t i = 0; i < op_conf.boxing_conf().out_num(); ++i) {
    EnrollOutputBn("out_" + std::to_string(i));
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

void BoxingOp::InferBlobDesc4FwBlobs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) const {
  // check boxing conf
  auto conf = op_conf().boxing_conf();
  auto in_box_case = conf.in_box_case();
  std::vector<int64_t> data_tmp_blob_shape_vec =
      GetBlobDesc4BnInOp(input_bns().at(0))->shape().dim_vec();
  CHECK_EQ(conf.in().data_type(), JobDesc::Singleton()->default_data_type());
  CHECK_EQ(conf.out().data_type(), JobDesc::Singleton()->default_data_type());
  int32_t concat_axis = 0;
  if (in_box_case == BoxingOpConf::kConcatBox) {
    concat_axis = conf.concat_box().axis();
    CHECK(concat_axis == 0 || concat_axis == 1);
  }

  // check datatype of input desc && calculate the shape of data_tmp
  for (size_t ib_idx = 1; ib_idx < input_bns().size(); ++ib_idx) {
    const BlobDesc* ib_desc = GetBlobDesc4BnInOp(input_bns().at(ib_idx));
    CHECK_EQ(ib_desc->data_type(), conf.in().data_type());
    auto ib_shape_vec = ib_desc->shape().dim_vec();
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

  // Although the shape of data_tmp is caculated in all kinds of concat boxes,
  // it is stored back if and only if this is a concat-clone box
  auto out_box_case = conf.out_box_case();
  CHECK_NE(out_box_case, BoxingOpConf::OUT_BOX_NOT_SET);
  const bool has_data_id = GetBlobDesc4BnInOp(input_bns().at(0))->has_data_id();
  if (in_box_case == BoxingOpConf::kConcatBox
      && out_box_case == BoxingOpConf::kCloneBox) {
    BlobDesc* dtb_desc = GetBlobDesc4BnInOp(SoleDtbn());
    dtb_desc->set_has_data_id(has_data_id);
    dtb_desc->mut_shape() = Shape(data_tmp_blob_shape_vec);
  }

  // infer desc of out blobs
  if (out_box_case == BoxingOpConf::kDataSplitBox) {
    int32_t out_num = output_bns().size();
    BalancedSplitter splitter(data_tmp_blob_shape_vec[0], out_num);
    auto output_shape_vec = data_tmp_blob_shape_vec;
    for (size_t i = 0; i < out_num; ++i) {
      BlobDesc* ob_desc = GetBlobDesc4BnInOp(output_bns().at(i));
      ob_desc->set_data_type(conf.out().data_type());
      ob_desc->set_has_data_id(has_data_id);
      output_shape_vec[0] = splitter.At(i).size();
      ob_desc->mut_shape() = Shape(output_shape_vec);
    }
  } else if (out_box_case == BoxingOpConf::kCloneBox) {
    for (auto obn : output_bns()) {
      GetBlobDesc4BnInOp(obn)->mut_shape() = Shape(data_tmp_blob_shape_vec);
    }
  } else {
    UNEXPECTED_RUN();
  }
}

REGISTER_OP(OperatorConf::kBoxingConf, BoxingOp);

}  // namespace oneflow
