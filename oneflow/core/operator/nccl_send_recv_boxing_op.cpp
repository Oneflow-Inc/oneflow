/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/operator/nccl_send_recv_boxing_op_util.h"

namespace oneflow {

class NcclSendRecvBoxingOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclSendRecvBoxingOp);
  NcclSendRecvBoxingOp() = default;
  ~NcclSendRecvBoxingOp() override = default;

  Maybe<void> InitFromOpConf() override;
  Maybe<void> InferInternalBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx, const JobDesc* job_desc) const override;
  Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    UNIMPLEMENTED_THEN_RETURN();
  }
  Maybe<void> InferOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;

 private:
  LogicalBlobId lbi4ibn(const std::string& input_bn) const override;
  LogicalBlobId lbi4obn(const std::string& output_bn) const override;
};

Maybe<void> NcclSendRecvBoxingOp::InitFromOpConf() {
  const NcclSendRecvBoxingOpConf& conf = this->op_conf().nccl_send_recv_boxing_conf();
  if (conf.has_input()) { EnrollInputBn("in", false); }
  if (conf.has_output()) { EnrollOutputBn("out", false); }
  EnrollTmpBn("buf");
  return Maybe<void>::Ok();
}

Maybe<void> NcclSendRecvBoxingOp::InferInternalBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const JobDesc* job_desc) const {
  BlobDesc* buf = GetBlobDesc4BnInOp("buf");
  const NcclSendRecvBoxingOpConf& conf = this->op_conf().nccl_send_recv_boxing_conf();
  const NdSbp& src_nd_sbp = conf.src_nd_sbp();
  const NdSbp& dst_nd_sbp = conf.dst_nd_sbp();
  ParallelDesc parallel_desc(conf.parallel_conf());
  ParallelDesc in_parallel_desc(conf.src_parallel_conf());
  ParallelDesc out_parallel_desc(conf.dst_parallel_conf());
  const int64_t parallel_num = parallel_desc.parallel_num();
  const int64_t parallel_id = parallel_ctx->parallel_id();
  const Shape& logical_shape = Shape(conf.logical_shape());
  std::vector<TensorSliceView> src_send_intersections;
  std::vector<TensorSliceView> dst_recv_intersections;
  GetRankSendRecvIntersection(parallel_id, parallel_desc, in_parallel_desc, out_parallel_desc,
                              src_nd_sbp, dst_nd_sbp, logical_shape, &src_send_intersections,
                              &dst_recv_intersections);
  int64_t buf_count = 0;
  if (conf.has_input()) {
    const BlobDesc* in = GetBlobDesc4BnInOp("in");
    buf->set_data_type(in->data_type());
    CHECK_EQ(src_send_intersections.size(), parallel_num);
    for (int64_t i = 0; i < parallel_num; ++i) {
      const TensorSliceView& intersection = JUST(VectorAt(src_send_intersections, i));
      if (!intersection.IsEmpty()) { buf_count += intersection.shape().elem_cnt(); }
    }
  }
  if (conf.has_output()) {
    const BlobDesc* out = GetBlobDesc4BnInOp("out");
    buf->set_data_type(out->data_type());
    for (int64_t i = 0; i < parallel_num; ++i) {
      const TensorSliceView& intersection = JUST(VectorAt(dst_recv_intersections, i));
      if (!intersection.IsEmpty()) { buf_count += intersection.shape().elem_cnt(); }
    }
    if (NdSbpHasPartialParallel(src_nd_sbp)) {
      // Note: when src_nd_sbp has partial_sum, need a out_size buffer to copy and add to out.
      buf_count += out->shape().elem_cnt();
    }
  }
  buf->set_shape(Shape({buf_count}));
  return Maybe<void>::Ok();
}

LogicalBlobId NcclSendRecvBoxingOp::lbi4ibn(const std::string& input_bn) const {
  return this->op_conf().nccl_send_recv_boxing_conf().lbi();
}

LogicalBlobId NcclSendRecvBoxingOp::lbi4obn(const std::string& output_bn) const {
  return this->op_conf().nccl_send_recv_boxing_conf().lbi();
}

Maybe<void> NcclSendRecvBoxingOp::InferOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const NcclSendRecvBoxingOpConf& conf = this->op_conf().nccl_send_recv_boxing_conf();
  const Shape& logical_shape = Shape(conf.logical_shape());
  const ParallelDesc& parallel_desc = ParallelDesc(conf.parallel_conf());
  const int64_t machine_id = JUST(parallel_desc.MachineId4ParallelId(parallel_ctx->parallel_id()));
  const int64_t device_index = JUST(parallel_desc.DeviceId4ParallelId(parallel_ctx->parallel_id()));
  if (conf.has_input()) {
    const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
    const NdSbp& src_nd_sbp = conf.src_nd_sbp();
    const ParallelDesc& src_parallel_desc = ParallelDesc(conf.src_parallel_conf());
    int64_t src_parallel_id =
        JUST(src_parallel_desc.ParallelId4MachineDeviceId(machine_id, device_index));
    std::shared_ptr<Shape> in_shape =
        JUST(GetPhysicalShape(logical_shape, src_nd_sbp, src_parallel_desc, src_parallel_id));
    CHECK_EQ_OR_RETURN(*in_shape, in_blob_desc->shape())
        << "Non-matching shape of blobs for pieces of nccl send recv";
  }
  if (conf.has_output()) {
    BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
    const NdSbp& dst_nd_sbp = conf.dst_nd_sbp();
    const ParallelDesc& dst_parallel_desc = ParallelDesc(conf.dst_parallel_conf());
    int64_t dst_parallel_id =
        JUST(dst_parallel_desc.ParallelId4MachineDeviceId(machine_id, device_index));
    std::shared_ptr<Shape> out_shape =
        JUST(GetPhysicalShape(logical_shape, dst_nd_sbp, dst_parallel_desc, dst_parallel_id));
    out_blob_desc->set_shape(*out_shape);
    out_blob_desc->set_data_type(conf.data_type());
  }
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kNcclSendRecvBoxingConf, NcclSendRecvBoxingOp);

}  // namespace oneflow
