#include "oneflow/core/operator/operator.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/kernel/cuda_ring_boxing_kernel_util.h"

namespace oneflow {

#ifdef WITH_CUDA

namespace {

class CudaRingAllReduceOpCtx final : public OpContext {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaRingAllReduceOpCtx);
  CudaRingAllReduceOpCtx(int64_t num_link, int64_t slice_factor, int64_t num_rank, int64_t num_elem,
                         DataType data_type)
      : num_link_(num_link), slice_factor_(slice_factor), num_rank_(num_rank) {
    const size_t pack_region_size = GetCudaRingBoxingPackCoalesceRegionSize();
    const size_t size_of_data_type = GetSizeOfDataType(data_type);
    CHECK_EQ(pack_region_size % size_of_data_type, 0);
    const int64_t num_elem_per_pack_region = pack_region_size / size_of_data_type;
    const int64_t num_pack_region =
        RoundUp(num_elem, num_elem_per_pack_region) / num_elem_per_pack_region;
    CHECK_GE(num_link, 1);
    const int64_t num_slice = num_link * slice_factor * num_rank;
    const int64_t num_pack_region_per_slice = RoundUp(num_pack_region, num_slice) / num_slice;
    const int64_t num_elem_per_slice = num_pack_region_per_slice * num_elem_per_pack_region;
    slices_.resize(num_slice);
    FOR_RANGE(int64_t, i, 0, num_slice) {
      slices_[i].mut_begin() = std::min(num_elem, i * num_elem_per_slice);
      slices_[i].mut_end() = std::min(num_elem, (i + 1) * num_elem_per_slice);
    }
  }

  ~CudaRingAllReduceOpCtx() override = default;

  const Range& GetSlice(int64_t link_id, int64_t slice_id, int64_t rank_id) const {
    CHECK_LT(link_id, num_link_);
    CHECK_LT(slice_id, slice_factor_);
    CHECK_LT(rank_id, num_rank_);
    return slices_.at(link_id * slice_factor_ * num_rank_ + slice_id * num_rank_ + rank_id);
  }

  void ForEachSlice(const std::function<void(int64_t link_id, int64_t slice_id, int64_t rank_id,
                                             const Range& range)>& Handler) const {
    FOR_RANGE(int64_t, link_id_, 0, num_link_) {
      FOR_RANGE(int64_t, slice_id, 0, slice_factor_) {
        FOR_RANGE(int64_t, rank_id, 0, num_rank_) {
          Handler(link_id_, slice_id, rank_id, GetSlice(link_id_, slice_id, rank_id));
        }
      }
    }
  }

  int64_t num_rank() const { return num_rank_; }

  int64_t num_link() const { return num_link_; }

  int64_t slice_factor() const { return slice_factor_; }

 private:
  int64_t num_link_;
  int64_t slice_factor_;
  int64_t num_rank_;
  std::vector<Range> slices_;
};

}  // namespace

class CudaRingAllReduceOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaRingAllReduceOp);
  CudaRingAllReduceOp() = default;
  ~CudaRingAllReduceOp() override = default;

  void InitFromOpConf() override;

 private:
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override;
  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
  const PbMessage& GetCustomizedConf() const override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx, int64_t record_piece_size,
                      std::function<void(OpContext*)> EnrollOpCtx) const override;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf*, const OpContext*) const override;
};

void CudaRingAllReduceOp::InitFromOpConf() {
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
  const int64_t num_link = op_conf().cuda_ring_all_reduce_conf().link_size();
  CHECK_GE(num_link, 1);
  EnrollRepeatedInputBn("recv", num_link, false);
  EnrollRepeatedOutputBn("send", num_link, false);
}

LogicalBlobId CudaRingAllReduceOp::ibn2lbi(const std::string& input_bn) const {
  return op_conf().cuda_ring_all_reduce_conf().lbi();
}

LogicalBlobId CudaRingAllReduceOp::obn2lbi(const std::string& output_bn) const {
  return op_conf().cuda_ring_all_reduce_conf().lbi();
}

const PbMessage& CudaRingAllReduceOp::GetCustomizedConf() const {
  return op_conf().cuda_ring_all_reduce_conf();
}

void CudaRingAllReduceOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, int64_t record_piece_size,
    std::function<void(OpContext*)> EnrollOpCtx) const {
  const CudaRingAllReduceOpConf& conf = op_conf().cuda_ring_all_reduce_conf();
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *in;
  const int64_t num_link = conf.link().size();
  const int64_t num_rank = conf.link(0).next_size();
  CHECK_EQ(parallel_ctx->parallel_num(), num_rank);
  FOR_RANGE(int64_t, i, 1, num_link) { CHECK_EQ(conf.link().Get(i).next_size(), num_rank); }
  const int64_t slice_factor = conf.slice_factor();
  auto* op_ctx = new CudaRingAllReduceOpCtx(num_link, slice_factor, num_rank,
                                            in->shape().elem_cnt(), in->data_type());
  EnrollOpCtx(op_ctx);
  std::vector<int64_t> max_buffer_size(num_link, 0);
  op_ctx->ForEachSlice([&max_buffer_size](const int64_t link_id, const int64_t slice_id,
                                          const int64_t rank_id, const Range& range) {
    max_buffer_size[link_id] = std::max(max_buffer_size.at(link_id), range.size());
  });
  FOR_RANGE(int64_t, i, 0, num_link) {
    BlobDesc* send_i = GetBlobDesc4BnInOp(GenRepeatedBn("send", i));
    send_i->set_data_type(in->data_type());
    send_i->mut_shape() = Shape({max_buffer_size.at(i)});
  }
}

void CudaRingAllReduceOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext* op_ctx) const {
  const auto* all_reduce_op_ctx = dynamic_cast<const CudaRingAllReduceOpCtx*>(op_ctx);
  CHECK_NOTNULL(all_reduce_op_ctx);
  const CudaRingAllReduceOpConf& conf = this->op_conf().cuda_ring_all_reduce_conf();
  CudaRingAllReduceKernelConf* cuda_ring_all_reduce_kernel_conf =
      kernel_conf->mutable_cuda_ring_all_reduce_conf();
  const int64_t num_link = all_reduce_op_ctx->num_link();
  CHECK_EQ(conf.link_size(), num_link);
  const int64_t num_rank = all_reduce_op_ctx->num_rank();
  CHECK_EQ(num_rank, parallel_ctx->parallel_num());
  const int64_t num_step = all_reduce_op_ctx->num_rank() * 2 - 1;
  cuda_ring_all_reduce_kernel_conf->set_num_link(num_link);
  cuda_ring_all_reduce_kernel_conf->set_num_step(num_step);
  cuda_ring_all_reduce_kernel_conf->set_slice_factor(all_reduce_op_ctx->slice_factor());
  std::vector<std::vector<int64_t>> link2step_rank_id(num_link);
  FOR_RANGE(int64_t, link_id, 0, num_link) {
    const CudaRingAllReduceLinkConf& link = conf.link(link_id);
    CHECK_EQ(link.next_size(), num_rank);
    link2step_rank_id.at(link_id).resize(num_step);
    std::vector<int64_t> link_prev(num_rank);
    FOR_RANGE(int64_t, i, 0, num_rank) {
      const int64_t next = link.next(i);
      CHECK_GE(next, 0);
      CHECK_LT(next, num_rank);
      link_prev[next] = i;
    }
    int64_t current_rank_id = parallel_ctx->parallel_id();
    FOR_RANGE(int64_t, step_id, 0, num_step) {
      current_rank_id = link_prev[current_rank_id];
      link2step_rank_id.at(link_id)[step_id] = current_rank_id;
    }
  }
  FOR_RANGE(int64_t, step_id, 0, num_step) {
    CudaRingAllReduceStepConf* step_conf =
        cuda_ring_all_reduce_kernel_conf->mutable_step_conf()->Add();
    if (step_id == 0) {
      step_conf->set_send(true);
      step_conf->set_recv(false);
      step_conf->set_in(true);
      step_conf->set_out(false);
    } else if (step_id < num_rank - 1) {
      step_conf->set_send(true);
      step_conf->set_recv(true);
      step_conf->set_in(true);
      step_conf->set_out(false);
    } else if (step_id == num_rank - 1) {
      step_conf->set_send(true);
      step_conf->set_recv(true);
      step_conf->set_in(true);
      step_conf->set_out(true);
    } else if (step_id < num_step - 1) {
      step_conf->set_send(true);
      step_conf->set_recv(true);
      step_conf->set_in(false);
      step_conf->set_out(true);
    } else if (step_id == num_step - 1) {
      step_conf->set_send(false);
      step_conf->set_recv(true);
      step_conf->set_in(false);
      step_conf->set_out(true);
    } else {
      UNIMPLEMENTED();
    }
    FOR_RANGE(int64_t, link_id, 0, all_reduce_op_ctx->num_link()) {
      CudaRingAllReduceStepLinkConf* link_conf = step_conf->mutable_link_conf()->Add();
      FOR_RANGE(int64_t, slice_id, 0, all_reduce_op_ctx->slice_factor()) {
        all_reduce_op_ctx->GetSlice(link_id, slice_id, link2step_rank_id.at(link_id).at(step_id))
            .ToProto(link_conf->mutable_slice_range()->Add());
      }
    }
  }
}

REGISTER_OP(OperatorConf::kCudaRingAllReduceConf, CudaRingAllReduceOp);

#endif

}  // namespace oneflow
