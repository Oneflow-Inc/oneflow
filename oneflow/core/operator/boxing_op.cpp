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

Symbol<OperatorConf> BoxingOp::GetOpConfWithoutOpNameAndLbn() const {
  OperatorConf op_conf(this->op_conf());
  op_conf.set_name("");
  CHECK(op_conf.has_boxing_conf());
  auto* boxing_conf = op_conf.mutable_boxing_conf();
  LogicalBlobId empty_logical_blob_id;
  *boxing_conf->mutable_lbi() = empty_logical_blob_id;
  return SymbolOf(op_conf);
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

  DimVector data_tmp_blob_shape_vec = GetBlobDesc4BnInOp(input_bns().Get(0))->shape().dim_vec();
  InferTmpBlobDesc(GetBlobDesc4BnInOp, &data_tmp_blob_shape_vec);

  if (conf.out_box_case() == BoxingOpConf::kSplitBox) {
    const BoxSplitConf& split_conf = conf.split_box();
    CHECK_GE_OR_RETURN(split_conf.axis(), 0);
    CHECK_LT_OR_RETURN(split_conf.axis(), data_tmp_blob_shape_vec.size());
    FOR_RANGE(size_t, i, 0, output_bns().size()) {
      BlobDesc* out_blob_desc = GetBlobDesc4BnInOp(output_bns().Get(i));
      *out_blob_desc = *first_in_blob;
      CHECK_GT_OR_RETURN(split_conf.part_num(i), 0);
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
    UNIMPLEMENTED_THEN_RETURN();
  }
  return Maybe<void>::Ok();
}

Maybe<void> BoxingOp::InferTmpBlobDesc(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    DimVector* data_tmp_vec_ptr) const {
  const BoxingOpConf& conf = op_conf().boxing_conf();
  if (conf.in_box_case() == BoxingOpConf::kConcatBox) {
    int32_t concat_axis = conf.concat_box().axis();
    CHECK_GE_OR_RETURN(concat_axis, 0);
    FOR_RANGE(size_t, ib_idx, 1, input_bns().size()) {
      const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp(input_bns().Get(ib_idx));
      const DimVector& in_blob_shape_vec = in_blob_desc->shape().dim_vec();
      CHECK_LT_OR_RETURN(concat_axis, in_blob_shape_vec.size());
      FOR_RANGE(size_t, i, 0, in_blob_shape_vec.size()) {
        if (i == concat_axis) {
          (*data_tmp_vec_ptr)[i] += in_blob_shape_vec[i];
        } else {
          CHECK_EQ_OR_RETURN((*data_tmp_vec_ptr)[i], in_blob_shape_vec[i]);
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
  }
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kBoxingConf, BoxingOp);

}  // namespace oneflow
