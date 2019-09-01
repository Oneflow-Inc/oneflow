#include "oneflow/core/operator/boxing_op.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

void BoxingOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  OpAttribute* op_attribute = kernel_conf->mutable_op_attribute();
  EraseEmptyBnInVec(GetBlobDesc4BnInOp, op_attribute->mutable_input_bns());
  EraseEmptyBnInVec(GetBlobDesc4BnInOp, op_attribute->mutable_output_bns());
}

void BoxingOp::InitFromOpConf() {
  CHECK(op_conf().has_boxing_conf());
  const BoxingOpConf& boxing_conf = op_conf().boxing_conf();

  for (int32_t i = 0; i < boxing_conf.in_num(); ++i) {
    EnrollInputBn("in_" + std::to_string(i), false);
  }
  if (boxing_conf.in_box_case() == BoxingOpConf::kAddBox
      && boxing_conf.out_box_case() == BoxingOpConf::kSplitBox) {
    EnrollTmpBn("middle");
  }
  for (int32_t i = 0; i < boxing_conf.out_num(); ++i) {
    EnrollOutputBn("out_" + std::to_string(i), false);
  }
}

const PbMessage& BoxingOp::GetCustomizedConf() const { return op_conf().boxing_conf(); }

LogicalBlobId BoxingOp::ibn2lbi(const std::string& input_bn) const {
  return GetMsgFromCustomizedConf<LogicalBlobId>("lbi");
}

LogicalBlobId BoxingOp::obn2lbi(const std::string& output_bn) const {
  return GetMsgFromCustomizedConf<LogicalBlobId>("lbi");
}

Maybe<void> BoxingOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BoxingOpConf& conf = op_conf().boxing_conf();
  BlobDesc* first_in_blob = GetBlobDesc4BnInOp(input_bns().Get(0));
  if (conf.in_box_case() == BoxingOpConf::kAddBox) {
    const Shape& first_in_blob_shape = first_in_blob->shape();
    for (const std::string& ibn : input_bns()) {
      CHECK_EQ_OR_RETURN(first_in_blob_shape, GetBlobDesc4BnInOp(ibn)->shape());
    }
  }

  std::vector<int64_t> data_tmp_blob_dim0_inner_shape_vec;
  if (GetBlobDesc4BnInOp(input_bns().Get(0))->has_dim0_inner_shape()) {
    data_tmp_blob_dim0_inner_shape_vec =
        GetBlobDesc4BnInOp(input_bns().Get(0))->dim0_inner_shape().dim_vec();
  }
  std::vector<int64_t> data_tmp_blob_shape_vec =
      GetBlobDesc4BnInOp(input_bns().Get(0))->shape().dim_vec();
  InferTmpBlobDesc(GetBlobDesc4BnInOp, &data_tmp_blob_shape_vec,
                   &data_tmp_blob_dim0_inner_shape_vec);

  if (conf.out_box_case() == BoxingOpConf::kSplitBox) {
    const BoxSplitConf& split_conf = conf.split_box();
    CHECK_GE_OR_RETURN(split_conf.axis(), 0);
    CHECK_LT_OR_RETURN(split_conf.axis(), data_tmp_blob_shape_vec.size());
    FOR_RANGE(size_t, i, 0, output_bns().size()) {
      BlobDesc* out_blob_desc = GetBlobDesc4BnInOp(output_bns().Get(i));
      *out_blob_desc = *first_in_blob;
      data_tmp_blob_shape_vec[split_conf.axis()] = split_conf.part_num(i);
      out_blob_desc->mut_shape() = Shape(data_tmp_blob_shape_vec);
      if (split_conf.axis() == 0 && data_tmp_blob_dim0_inner_shape_vec.size() > 0) {
        size_t inner_shape_count_1 = Shape(data_tmp_blob_dim0_inner_shape_vec).Count(1);
        CHECK_EQ_OR_RETURN(split_conf.part_num(i) % inner_shape_count_1, 0);
        data_tmp_blob_dim0_inner_shape_vec[0] = split_conf.part_num(i) / inner_shape_count_1;
        out_blob_desc->mut_dim0_inner_shape() = Shape(data_tmp_blob_dim0_inner_shape_vec);
      }
    }
  } else if (conf.out_box_case() == BoxingOpConf::kCloneBox) {
    for (const std::string& obn : output_bns()) {
      BlobDesc* out_blob_desc = GetBlobDesc4BnInOp(obn);
      *out_blob_desc = *first_in_blob;
      out_blob_desc->mut_shape() = Shape(data_tmp_blob_shape_vec);
      if (data_tmp_blob_dim0_inner_shape_vec.size() > 0) {
        out_blob_desc->mut_dim0_inner_shape() = Shape(data_tmp_blob_dim0_inner_shape_vec);
      }
    }
  } else {
    UNIMPLEMENTED_THEN_RETURN();
  }
  return Maybe<void>::Ok();
}

Maybe<void> BoxingOp::InferTmpBlobDesc(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    std::vector<int64_t>* data_tmp_vec_ptr,
    std::vector<int64_t>* data_tmp_dim0_inner_shape_vec_ptr) const {
  const BoxingOpConf& conf = op_conf().boxing_conf();
  if (conf.in_box_case() == BoxingOpConf::kConcatBox) {
    int32_t concat_axis = conf.concat_box().axis();
    CHECK_GE_OR_RETURN(concat_axis, 0);
    FOR_RANGE(size_t, ib_idx, 1, input_bns().size()) {
      const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp(input_bns().Get(ib_idx));
      const std::vector<int64_t>& in_blob_shape_vec = in_blob_desc->shape().dim_vec();
      const std::vector<int64_t>* in_blob_dim0_inner_shape_vec = nullptr;
      if (in_blob_desc->has_dim0_inner_shape()) {
        in_blob_dim0_inner_shape_vec = &in_blob_desc->dim0_inner_shape().dim_vec();
      }
      CHECK_LT_OR_RETURN(concat_axis, in_blob_shape_vec.size());
      FOR_RANGE(size_t, i, 0, in_blob_shape_vec.size()) {
        if (i == concat_axis) {
          (*data_tmp_vec_ptr)[i] += in_blob_shape_vec[i];
          if (i == 0 && in_blob_dim0_inner_shape_vec) {
            (*data_tmp_dim0_inner_shape_vec_ptr)[0] += (*in_blob_dim0_inner_shape_vec)[0];
          }
        } else {
          CHECK_EQ_OR_RETURN((*data_tmp_vec_ptr)[i], in_blob_shape_vec[i]);
          if (i == 0 && in_blob_dim0_inner_shape_vec) {
            CHECK_OR_RETURN(*data_tmp_dim0_inner_shape_vec_ptr == *in_blob_dim0_inner_shape_vec);
          }
        }
      }
    }
  }

  CHECK_NE_OR_RETURN(conf.out_box_case(), BoxingOpConf::OUT_BOX_NOT_SET);
  if (conf.in_box_case() == BoxingOpConf::kAddBox
      && conf.out_box_case() == BoxingOpConf::kSplitBox) {
    BlobDesc* data_tmp_blob_desc = GetBlobDesc4BnInOp(SoleTbn());
    data_tmp_blob_desc->mut_shape() = Shape(*data_tmp_vec_ptr);
    data_tmp_blob_desc->set_data_type(GetBlobDesc4BnInOp(input_bns().Get(0))->data_type());
    if (data_tmp_dim0_inner_shape_vec_ptr->size() > 0) {
      data_tmp_blob_desc->mut_dim0_inner_shape() = Shape(*data_tmp_dim0_inner_shape_vec_ptr);
    }
  }
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kBoxingConf, BoxingOp);

}  // namespace oneflow
