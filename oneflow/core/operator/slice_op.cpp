#include "oneflow/core/operator/slice_op.h"

namespace oneflow {

void SliceOp::InitFromOpConf() {
  CHECK(op_conf().has_slice_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
  if (op_conf().device_type() == DeviceType::kGPU) {
    EnrollConstBufBn("slice_dim_index");
    EnrollConstBufBn("slice_dim_elem_cnt");
  }
}

const PbMessage& SliceOp::GetCustomizedConf() const { return op_conf().slice_conf(); }

void SliceOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const {
  const SliceOpConf& conf = op_conf().slice_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_EQ(conf.dim_slice_conf_size(), in_blob_desc->shape().NumAxes() - 1);
  std::vector<int64_t> shape_vec(in_blob_desc->shape().NumAxes());
  shape_vec[0] = in_blob_desc->shape().At(0);
  FOR_RANGE(size_t, i, 0, conf.dim_slice_conf_size()) {
    int32_t dim_len = in_blob_desc->shape().At(i + 1);
    const DimSliceConf& dim_slice_conf = conf.dim_slice_conf(i);
    int32_t step = dim_slice_conf.stride();
    CHECK_GT(step, 0);
    int32_t start = dim_slice_conf.has_start() ? dim_slice_conf.start() : 0;
    int32_t end = dim_slice_conf.has_end() ? dim_slice_conf.end() : dim_len;
    if (start < 0) { start += dim_len; }
    if (end < 0) { end += dim_len; }
    CHECK_GE(start, 0);
    CHECK_LT(start, end);
    CHECK_LE(end, dim_len);
    shape_vec[i + 1] = (end - start - 1) / std::abs(step) + 1;
  }

  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  out_blob_desc->mut_shape() = Shape(shape_vec);

  BlobDesc* dim_index_blob_desc = GetBlobDesc4BnInOp("slice_dim_index");
  if (dim_index_blob_desc) {
    dim_index_blob_desc->mut_shape() =
        Shape({out_blob_desc->shape().elem_cnt(), out_blob_desc->shape().NumAxes()});
    dim_index_blob_desc->set_data_type(DataType::kInt32);
  }
  BlobDesc* dim_elem_cnt_blob_desc = GetBlobDesc4BnInOp("slice_dim_elem_cnt");
  if (dim_elem_cnt_blob_desc) {
    dim_elem_cnt_blob_desc->mut_shape() = Shape({out_blob_desc->shape().NumAxes()});
    dim_elem_cnt_blob_desc->set_data_type(DataType::kInt32);
  }
}

REGISTER_OP(OperatorConf::kSliceConf, SliceOp);

}  // namespace oneflow
