#include "oneflow/core/operator/boxing_op.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

LogicalBlobId BoxingOp::ibn2lbi(const std::string& input_bn) const { return boxing_conf().lbi(); }
LogicalBlobId BoxingOp::obn2lbi(const std::string& output_bn) const { return boxing_conf().lbi(); }

std::vector<int64_t> BoxingOp::CalcDataTmpBlobShapeVec(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const PbRpf<std::string>& input_bns) const {
  const BoxingOpConf conf = boxing_conf();
  std::vector<int64_t> data_tmp_blob_shape_vec =
      GetBlobDesc4BnInOp(input_bns.Get(0))->shape().dim_vec();
  if (conf.in_box_case() == BoxingOpConf::kConcatBox) {
    int32_t concat_axis = conf.concat_box().axis();
    CHECK_GE(concat_axis, 0);
    FOR_RANGE(size_t, ib_idx, 1, input_bns.size()) {
      const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp(input_bns.Get(ib_idx));
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

}  // namespace oneflow
