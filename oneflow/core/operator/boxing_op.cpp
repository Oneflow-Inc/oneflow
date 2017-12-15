#include "oneflow/core/operator/boxing_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

namespace {

void EraseEmptyBnInVec(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    PbRpf<std::string>* bns) {
  for (auto it = bns->begin(); it != bns->end();) {
    if (!GetBlobDesc4BnInOp(*it)) {
      it = bns->erase(it);
    } else {
      ++it;
    }
  }
}

#define ERASE_BNS(bns) EraseEmptyBnInVec(GetBlobDesc4BnInOp, bns);

}  // namespace

void BoxingOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  ERASE_BNS(kernel_conf->mutable_input_bns());
  ERASE_BNS(kernel_conf->mutable_output_bns());
}

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
