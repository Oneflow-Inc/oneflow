#include "oneflow/core/operator/boxing_op.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

LogicalBlobId BoxingOp::ibn2lbi(const std::string& input_bn) const { return boxing_conf().lbi(); }
LogicalBlobId BoxingOp::obn2lbi(const std::string& output_bn) const { return boxing_conf().lbi(); }

std::vector<int64_t> BoxingOp::CalcDataTmpBlobShapeVec(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp) const {
  const BoxingOpConf conf = boxing_conf();
  std::vector<int64_t> data_tmp_blob_shape_vec =
      GetBlobDesc4BnInOp(InputBns().Get(0))->shape().dim_vec();
  if (conf.in_box_case() == BoxingOpConf::kConcatBox) {
    int32_t concat_axis = conf.concat_box().axis();
    CHECK_GE(concat_axis, 0);
    FOR_RANGE(size_t, ib_idx, 1, InputBns().size()) {
      const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp(InputBns().Get(ib_idx));
      const std::vector<int64_t>& in_blob_shape_vec = in_blob_desc->shape().dim_vec();
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
  return data_tmp_blob_shape_vec;
}

void BoxingOp::InferOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const std::vector<int64_t>& data_tmp_blob_shape_vec) const {
  const BoxingOpConf& conf = boxing_conf();
  const BlobDesc* first_in_blob = GetBlobDesc4BnInOp(InputBns().Get(0));
  CHECK_EQ(data_tmp_blob_shape_vec.at(0) % Global<JobDesc>::Get()->PieceSize(), 0);
  const size_t scale = data_tmp_blob_shape_vec.at(0) / Global<JobDesc>::Get()->PieceSize();
  if (conf.out_box_case() == BoxingOpConf::kSplitBox) {
    const BoxSplitConf& split_conf = conf.split_box();
    CHECK_GE(split_conf.axis(), 0);
    CHECK_LT(split_conf.axis(), data_tmp_blob_shape_vec.size());
    FOR_RANGE(size_t, i, 0, OutputBns().size()) {
      BlobDesc* out_blob_desc = GetBlobDesc4BnInOp(OutputBns().Get(i));
      *out_blob_desc = *first_in_blob;
      std::vector<int64_t> out_blob_shape_vec(data_tmp_blob_shape_vec);
      out_blob_shape_vec[split_conf.axis()] = split_conf.part_num(i) * scale;
      out_blob_desc->mut_shape() = Shape(out_blob_shape_vec);
    }
  } else if (conf.out_box_case() == BoxingOpConf::kCloneBox) {
    for (const std::string& obn : OutputBns()) {
      BlobDesc* out_blob_desc = GetBlobDesc4BnInOp(obn);
      *out_blob_desc = *first_in_blob;
      out_blob_desc->mut_shape() = Shape(data_tmp_blob_shape_vec);
    }
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace oneflow
