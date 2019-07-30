#include "oneflow/core/operator/cuda_ring_all_reduce_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/kernel/cuda_ring_boxing_kernel_util.h"

namespace oneflow {

namespace {

class DataContentSplitHelper final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataContentSplitHelper);
  DataContentSplitHelper(int64_t num_link, int64_t link_dup_factor, int64_t num_rank,
                         int64_t num_elem, DataType data_type)
      : num_link_(num_link), link_dup_factor_(link_dup_factor), num_rank_(num_rank) {
    const size_t align_size = GetCudaRingAllReducePackAlignSize();
    const size_t size_of_data_type = GetSizeOfDataType(data_type);
    CHECK_EQ(align_size % size_of_data_type, 0);
    const int64_t num_elem_per_pack = align_size / size_of_data_type;
    const int64_t num_pack = RoundUp(num_elem, num_elem_per_pack) / num_elem_per_pack;
    CHECK_GE(num_link, 1);
    const int64_t num_chunk = num_link * link_dup_factor * num_rank;
    const int64_t num_pack_per_chunk = RoundUp(num_pack, num_chunk) / num_chunk;
    const int64_t num_elem_per_chunk = num_pack_per_chunk * num_elem_per_pack;
    chunks_.resize(num_chunk);
    FOR_RANGE(int64_t, i, 0, num_chunk) {
      chunks_[i].mut_begin() = std::min(num_elem, i * num_elem_per_chunk);
      chunks_[i].mut_end() = std::min(num_elem, (i + 1) * num_elem_per_chunk);
    }
  }

  const Range& GetSplit(int64_t link_id, int64_t link_dup_id, int64_t rank_id) const {
    CHECK_LT(link_id, num_link_);
    CHECK_LT(link_dup_id, link_dup_factor_);
    CHECK_LT(rank_id, num_rank_);
    return chunks_.at(link_id * link_dup_factor_ * num_rank_ + link_dup_id * num_rank_ + rank_id);
  }

  void ForEachSplit(const std::function<void(int64_t link_id, int64_t link_dup_id, int64_t rank_id,
                                             const Range& range)>& Handler) const {
    FOR_RANGE(int64_t, link_id_, 0, num_link_) {
      FOR_RANGE(int64_t, link_dup_id, 0, link_dup_factor_) {
        FOR_RANGE(int64_t, rank_id, 0, num_rank_) {
          Handler(link_id_, link_dup_id, rank_id, GetSplit(link_id_, link_dup_id, rank_id));
        }
      }
    }
  }
  int64_t num_link_;
  int64_t link_dup_factor_;
  int64_t num_rank_;
  std::vector<Range> chunks_;
};

}  // namespace

void CudaRingAllReduceOp::InitFromOpConf() {
  const CudaRingAllReduceOpConf& conf = op_conf().cuda_ring_all_reduce_conf();
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
  const int64_t num_link = conf.link().size();
  CHECK_GE(num_link, 1);
  const int64_t num_rank = conf.link().Get(0).next_size();
  FOR_RANGE(int64_t, i, 1, num_link) { CHECK_EQ(conf.link().Get(i).next_size(), num_rank); }
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
    const ParallelContext* parallel_ctx) const {
  const CudaRingAllReduceOpConf& conf = op_conf().cuda_ring_all_reduce_conf();
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *in;
  const int64_t num_link = conf.link().size();
  std::vector<int64_t> max_buffer_size(num_link, 0);
  const int64_t num_rank = conf.link(0).next_size();
  const int64_t num_link_dup = conf.num_link_dup();
  const DataContentSplitHelper helper(num_link, num_link_dup, num_rank, in->shape().elem_cnt(),
                                      in->data_type());
  helper.ForEachSplit([&max_buffer_size](int64_t link_id, int64_t link_dup_id, int64_t rank_id,
                                         const Range& range) {
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
  const CudaRingAllReduceOpConf& conf = this->op_conf().cuda_ring_all_reduce_conf();
  CudaRingAllReduceKernelConf* multi_cuda_ring_all_reduce_kernel_conf =
      kernel_conf->mutable_cuda_ring_all_reduce_conf();
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  const int64_t num_link = conf.link().size();
  const int64_t num_rank = conf.link(0).next_size();
  const int64_t num_link_dup = conf.num_link_dup();
  const DataContentSplitHelper helper(num_link, num_link_dup, num_rank, in->shape().elem_cnt(),
                                      in->data_type());
  const int64_t num_step = num_rank * 2 - 1;
  multi_cuda_ring_all_reduce_kernel_conf->set_num_step(num_step);
  multi_cuda_ring_all_reduce_kernel_conf->set_num_link_dup(num_link_dup);
  std::vector<std::vector<int64_t>> step_rank_id;
  step_rank_id.resize(num_link);
  FOR_RANGE(int64_t, link_id, 0, num_link) {
    const RingLinkConf& link = conf.link(link_id);
    CHECK_EQ(link.next_size(), parallel_ctx->parallel_num());
    step_rank_id.at(link_id).resize(num_step);
    std::vector<int64_t> link_prev(link.next_size());
    FOR_RANGE(int64_t, i, 0, link.next_size()) { link_prev[link.next(i)] = i; }
    int64_t current_rank_id = parallel_ctx->parallel_id();
    FOR_RANGE(int64_t, step_id, 0, num_step) {
      current_rank_id = link_prev[current_rank_id];
      step_rank_id.at(link_id)[step_id] = current_rank_id;
    }
  }
  FOR_RANGE(int64_t, step_id, 0, num_step) {
    CudaRingAllReduceStepConf* step_conf =
        multi_cuda_ring_all_reduce_kernel_conf->mutable_step_conf()->Add();
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
    FOR_RANGE(int64_t, link_dup_id, 0, num_link_dup) {
      CudaRingAllReduceStepLinkConf* link_conf = step_conf->mutable_link_conf()->Add();
      FOR_RANGE(int64_t, link_id, 0, num_link) {
        helper.GetSplit(link_id, link_dup_id, step_rank_id.at(link_id).at(step_id))
            .ToProto(link_conf->mutable_link_data_range()->Add());
      }
    }
  }
}

REGISTER_OP(OperatorConf::kCudaRingAllReduceConf, CudaRingAllReduceOp);

}  // namespace oneflow
