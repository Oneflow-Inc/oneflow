#include "oneflow/core/operator/ring_boxing_op.h"
#include "oneflow/core/register/tensor_slice_view.h"

namespace oneflow {

void RingBoxingOp::InitFromOpConf() {
  EnrollInputBn("in", false);
  EnrollInputBn("recv", false);
  EnrollOutputBn("out", false);
  EnrollOutputBn("send", false);
}

LogicalBlobId RingBoxingOp::ibn2lbi(const std::string& input_bn) const {
  return GetCustomizedBoxingConf().lbi();
}

LogicalBlobId RingBoxingOp::obn2lbi(const std::string& output_bn) const {
  return GetCustomizedBoxingConf().lbi();
}

const RingBoxingConf& RingBoxingOp::GetCustomizedBoxingConf() const {
  return GetMsgFromCustomizedConf<RingBoxingConf>("ring_boxing_conf");
}

const PbMessage& RingReduceScatterOp::GetCustomizedConf() const {
  return op_conf().ring_reduce_scatter_conf();
}

void RingReduceScatterOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const RingBoxingConf& conf = GetCustomizedBoxingConf();
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  BlobDesc* send = GetBlobDesc4BnInOp("send");
  *out = *in;
  std::vector<TensorSliceView> slices(conf.slices_size());
  std::transform(conf.slices().cbegin(), conf.slices().cend(), slices.begin(),
                 [](const TensorSliceViewProto& proto) { return TensorSliceView(proto); });
  const TensorSliceView& out_slice = slices.at(parallel_ctx->parallel_id());
  if (conf.has_out_shape()) {
    const Shape out_shape(conf.out_shape());
    CHECK_EQ(out_slice.shape().elem_cnt(), out_shape.elem_cnt());
    out->mut_shape() = out_shape;
  } else {
    out->mut_shape() = out_slice.shape();
  }
  bool buffer_is_dynamic = false;
  int64_t max_elem_cnt = slices.front().shape().elem_cnt();
  FOR_RANGE(int64_t, i, 1, slices.size()) {
    int64_t elem_cnt = slices.at(i).shape().elem_cnt();
    if (elem_cnt != max_elem_cnt) {
      max_elem_cnt = std::max(max_elem_cnt, elem_cnt);
      buffer_is_dynamic = true;
    }
  }
  *send = *in;
  send->mut_shape() = Shape({max_elem_cnt});
  if (buffer_is_dynamic) {
    send->set_has_dim0_valid_num_field(true);
    send->mut_dim0_inner_shape() = Shape({1, max_elem_cnt});
  }
}

const PbMessage& RingAllGatherOp::GetCustomizedConf() const {
  return op_conf().ring_all_gather_conf();
}

void RingAllGatherOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const RingBoxingConf& conf = GetCustomizedBoxingConf();
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  BlobDesc* send = GetBlobDesc4BnInOp("send");
  *out = *in;
  const Shape logical_blob_shape(conf.logical_blob_shape());
  if (conf.has_out_shape()) {
    const Shape out_shape(conf.out_shape());
    CHECK_EQ(logical_blob_shape.elem_cnt(), out_shape.elem_cnt());
    out->mut_shape() = out_shape;
  } else {
    out->mut_shape() = logical_blob_shape;
  }
  std::vector<TensorSliceView> slices(conf.slices_size());
  std::transform(conf.slices().cbegin(), conf.slices().cend(), slices.begin(),
                 [](const TensorSliceViewProto& proto) { return TensorSliceView(proto); });
  bool buffer_is_dynamic = false;
  int64_t max_elem_cnt = slices.front().shape().elem_cnt();
  FOR_RANGE(int64_t, i, 1, slices.size()) {
    int64_t elem_cnt = slices.at(i).shape().elem_cnt();
    if (elem_cnt != max_elem_cnt) {
      max_elem_cnt = std::max(max_elem_cnt, elem_cnt);
      buffer_is_dynamic = true;
    }
  }
  *send = *in;
  send->mut_shape() = Shape({max_elem_cnt});
  if (buffer_is_dynamic) {
    send->set_has_dim0_valid_num_field(true);
    send->mut_dim0_inner_shape() = Shape({1, max_elem_cnt});
  }
}

const PbMessage& RingAllReduceOp::GetCustomizedConf() const {
  return op_conf().ring_all_reduce_conf();
}

void RingAllReduceOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const RingBoxingConf& conf = GetCustomizedBoxingConf();
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  BlobDesc* send = GetBlobDesc4BnInOp("send");
  *out = *in;
  const Shape logical_blob_shape(conf.logical_blob_shape());
  if (conf.has_out_shape()) {
    const Shape out_shape(conf.out_shape());
    CHECK_EQ(logical_blob_shape.elem_cnt(), out_shape.elem_cnt());
    out->mut_shape() = out_shape;
  } else {
    out->mut_shape() = logical_blob_shape;
  }
  std::vector<TensorSliceView> slices(conf.slices_size());
  std::transform(conf.slices().cbegin(), conf.slices().cend(), slices.begin(),
                 [](const TensorSliceViewProto& proto) { return TensorSliceView(proto); });
  bool buffer_is_dynamic = false;
  int64_t max_elem_cnt = slices.front().shape().elem_cnt();
  FOR_RANGE(int64_t, i, 1, slices.size()) {
    int64_t elem_cnt = slices.at(i).shape().elem_cnt();
    if (elem_cnt != max_elem_cnt) {
      max_elem_cnt = std::max(max_elem_cnt, elem_cnt);
      buffer_is_dynamic = true;
    }
  }
  *send = *in;
  send->mut_shape() = Shape({max_elem_cnt});
  if (buffer_is_dynamic) {
    send->set_has_dim0_valid_num_field(true);
    send->mut_dim0_inner_shape() = Shape({1, max_elem_cnt});
  }
}

REGISTER_OP(OperatorConf::kRingReduceScatterConf, RingReduceScatterOp);
REGISTER_OP(OperatorConf::kRingAllGatherConf, RingAllGatherOp);
REGISTER_OP(OperatorConf::kRingAllReduceConf, RingAllReduceOp);

}  // namespace oneflow
