#include "oneflow/core/operator/pod_boxing_op.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

void PodBoxingOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  CHECK(!boxing_conf().lbi().is_pb_blob());
  OpAttribute* op_attribute = kernel_conf->mutable_op_attribute();
  EraseEmptyBnInVec(GetBlobDesc4BnInOp, op_attribute->mutable_input_bns());
  EraseEmptyBnInVec(GetBlobDesc4BnInOp, op_attribute->mutable_output_bns());
}

void PodBoxingOp::InitFromOpConf() {
  CHECK(op_conf().has_pod_boxing_conf());

  for (int32_t i = 0; i < boxing_conf().in_num(); ++i) {
    EnrollInputBn("in_" + std::to_string(i), false);
  }
  if (boxing_conf().in_box_case() == BoxingOpConf::kAddBox
      && boxing_conf().out_box_case() == BoxingOpConf::kSplitBox) {
    EnrollDataTmpBn("middle");
  }
  for (int32_t i = 0; i < boxing_conf().out_num(); ++i) {
    EnrollOutputBn("out_" + std::to_string(i), false);
  }
}

const BoxingOpConf& PodBoxingOp::boxing_conf() const {
  return op_conf().pod_boxing_conf().boxing_conf();
}

void PodBoxingOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext* parallel_ctx) const {
  const BoxingOpConf& conf = boxing_conf();
  BlobDesc* first_in_blob = GetBlobDesc4BnInOp(input_bns().Get(0));
  if (conf.in_box_case() == BoxingOpConf::kAddBox) {
    const Shape& first_in_blob_shape = first_in_blob->shape();
    for (const std::string& ibn : input_bns()) {
      CHECK_EQ(first_in_blob_shape, GetBlobDesc4BnInOp(ibn)->shape());
    }
  }

  std::vector<int64_t> data_tmp_blob_shape_vec =
      GetBlobDesc4BnInOp(input_bns().Get(0))->shape().dim_vec();
  InferDataTmpBlobDesc(GetBlobDesc4BnInOp, &data_tmp_blob_shape_vec);

  if (conf.out_box_case() == BoxingOpConf::kSplitBox) {
    const BoxSplitConf& split_conf = conf.split_box();
    CHECK_GE(split_conf.axis(), 0);
    CHECK_LT(split_conf.axis(), data_tmp_blob_shape_vec.size());
    FOR_RANGE(size_t, i, 0, output_bns().size()) {
      BlobDesc* out_blob_desc = GetBlobDesc4BnInOp(output_bns().Get(i));
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

void PodBoxingOp::InferDataTmpBlobDesc(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    std::vector<int64_t>* data_tmp_vec_ptr) const {
  const BoxingOpConf& conf = boxing_conf();
  *data_tmp_vec_ptr = CalcDataTmpBlobShapeVec(GetBlobDesc4BnInOp, input_bns());
  CHECK_NE(conf.out_box_case(), BoxingOpConf::OUT_BOX_NOT_SET);
  if (conf.in_box_case() == BoxingOpConf::kAddBox
      && conf.out_box_case() == BoxingOpConf::kSplitBox) {
    BlobDesc* data_tmp_blob_desc = GetBlobDesc4BnInOp(SoleDtbn());
    data_tmp_blob_desc->mut_shape() = Shape(*data_tmp_vec_ptr);
    data_tmp_blob_desc->set_data_type(GetBlobDesc4BnInOp(input_bns().Get(0))->data_type());
  }
}

REGISTER_OP(OperatorConf::kPodBoxingConf, PodBoxingOp);

}  // namespace oneflow
