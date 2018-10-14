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
  BlobDesc* first_in_blob = GetBlobDesc4BnInOp(InputBns().Get(0));
  if (boxing_conf().in_box_case() == BoxingOpConf::kAddBox) {
    const Shape& first_in_blob_shape = first_in_blob->shape();
    for (const std::string& ibn : InputBns()) {
      CHECK_EQ(first_in_blob_shape, GetBlobDesc4BnInOp(ibn)->shape());
    }
  }

  std::vector<int64_t> dim0_inner_shape_vec;
  std::vector<int64_t> data_tmp_blob_shape_vec =
      GetBlobDesc4BnInOp(InputBns().Get(0))->shape().dim_vec();
  InferDataTmpBlobDesc(GetBlobDesc4BnInOp, &data_tmp_blob_shape_vec, &dim0_inner_shape_vec);
  InferOutBlobDescs(GetBlobDesc4BnInOp, data_tmp_blob_shape_vec, dim0_inner_shape_vec);
}

void PodBoxingOp::InferDataTmpBlobDesc(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    std::vector<int64_t>* data_tmp_vec_ptr, std::vector<int64_t>* dim0_inner_shape_vec) const {
  const BoxingOpConf& conf = boxing_conf();
  *data_tmp_vec_ptr = CalcDataTmpBlobShapeVec(GetBlobDesc4BnInOp, dim0_inner_shape_vec);
  CHECK_NE(conf.out_box_case(), BoxingOpConf::OUT_BOX_NOT_SET);
  if (conf.in_box_case() == BoxingOpConf::kAddBox
      && conf.out_box_case() == BoxingOpConf::kSplitBox) {
    BlobDesc* data_tmp_blob_desc = GetBlobDesc4BnInOp(SoleDtbn());
    data_tmp_blob_desc->mut_shape() = Shape(*data_tmp_vec_ptr);
    data_tmp_blob_desc->set_data_type(GetBlobDesc4BnInOp(InputBns().Get(0))->data_type());
    if (dim0_inner_shape_vec->size() > 0) {
      data_tmp_blob_desc->mut_dim0_inner_shape() = Shape(*dim0_inner_shape_vec);
    }
  }
}

REGISTER_OP(OperatorConf::kPodBoxingConf, PodBoxingOp);

}  // namespace oneflow
