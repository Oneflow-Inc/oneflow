#include "oneflow/core/operator/boxing_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

void BoxingOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  EraseEmptyBnInVec(GetBlobDesc4BnInOp, kernel_conf->mutable_input_bns());
  EraseEmptyBnInVec(GetBlobDesc4BnInOp, kernel_conf->mutable_output_bns());
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

const PbMessage& BoxingOp::GetCustomizedConf() const {
  return op_conf().boxing_conf();
}

std::string BoxingOp::ibn2lbn(const std::string& input_bn) const {
  return GetValFromCustomizedConf<std::string>("lbn");
}

std::string BoxingOp::obn2lbn(const std::string& output_bn) const {
  return GetValFromCustomizedConf<std::string>("lbn");
}

void BoxingOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BoxingOpConf& conf = op_conf().boxing_conf();
  BlobDesc* first_in_blob = GetBlobDesc4BnInOp(input_bns().front());
  if (conf.in_box_case() == BoxingOpConf::kAddBox) {
    const Shape& first_in_blob_shape = first_in_blob->shape();
    for (const std::string& ibn : input_bns()) {
      CHECK_EQ(first_in_blob_shape, GetBlobDesc4BnInOp(ibn)->shape());
    }
  }

  std::vector<int64_t> data_tmp_blob_shape_vec =
      GetBlobDesc4BnInOp(input_bns().front())->shape().dim_vec();
  InferDataTmpBlobDesc(GetBlobDesc4BnInOp, &data_tmp_blob_shape_vec);

  if (conf.out_box_case() == BoxingOpConf::kSplitBox) {
    const BoxSplitConf& split_conf = conf.split_box();
    CHECK_GE(split_conf.axis(), 0);
    CHECK_LT(split_conf.axis(), data_tmp_blob_shape_vec.size());
    FOR_RANGE(size_t, i, 0, output_bns().size()) {
      BlobDesc* out_blob_desc = GetBlobDesc4BnInOp(output_bns().at(i));
      *out_blob_desc = *first_in_blob;
      data_tmp_blob_shape_vec[split_conf.axis()] = split_conf.part_num(i);
      out_blob_desc->mut_shape() = Shape(data_tmp_blob_shape_vec);
    }
  } else if (conf.out_box_case() == BoxingOpConf::kCloneBox) {
    for (const std::string& obn : output_bns()) {
      BlobDesc* out_blob_desc = GetBlobDesc4BnInOp(obn);
      *out_blob_desc = *first_in_blob;
      out_blob_desc->mut_shape() = Shape(data_tmp_blob_shape_vec);
    }
  } else {
    UNIMPLEMENTED();
  }
}

void BoxingOp::InferDataTmpBlobDesc(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    std::vector<int64_t>* data_tmp_vec_ptr) const {
  const BoxingOpConf& conf = op_conf().boxing_conf();
  if (conf.in_box_case() == BoxingOpConf::kConcatBox) {
    int32_t concat_axis = conf.concat_box().axis();
    CHECK_GE(concat_axis, 0);
    FOR_RANGE(size_t, ib_idx, 1, input_bns().size()) {
      const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp(input_bns().at(ib_idx));
      const std::vector<int64_t>& in_blob_shape_vec =
          in_blob_desc->shape().dim_vec();
      CHECK_LT(concat_axis, in_blob_shape_vec.size());
      FOR_RANGE(size_t, i, 0, in_blob_shape_vec.size()) {
        if (i == concat_axis) {
          (*data_tmp_vec_ptr)[i] += in_blob_shape_vec[i];
        } else {
          CHECK_EQ((*data_tmp_vec_ptr)[i], in_blob_shape_vec[i]);
        }
      }
    }
  }

  CHECK_NE(conf.out_box_case(), BoxingOpConf::OUT_BOX_NOT_SET);
  if (conf.in_box_case() == BoxingOpConf::kAddBox
      && conf.out_box_case() == BoxingOpConf::kSplitBox) {
    BlobDesc* data_tmp_blob_desc = GetBlobDesc4BnInOp(SoleDtbn());
    data_tmp_blob_desc->mut_shape() = Shape(*data_tmp_vec_ptr);
    data_tmp_blob_desc->set_data_type(
        GetBlobDesc4BnInOp(input_bns().front())->data_type());
  }
}

REGISTER_OP(OperatorConf::kBoxingConf, BoxingOp);

}  // namespace oneflow
