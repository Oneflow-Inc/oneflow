#include "oneflow/core/data/dataset.h"

namespace oneflow {
namespace data {

DataSampler::DataSampler(Dataset* dataset)
    : dataset_(dataset), max_count_(0), gen_(dataset->dataset_proto().random_seed()) {}

void DataSampler::SubmitContext(DataSamplerContext* ctx) {
  size_t sample_count = dataset_->Size() / ctx->num_replicas_;
  size_t one_more_sample_rank = dataset_->Size() % ctx->num_replicas_;
  if (ctx->rank_ < one_more_sample_rank) { sample_count += 1; }
  std::unique_lock<std::mutex> lck(mtx_);
  max_count_ += sample_count;
}

void DataSampler::GenNewEpochIndexSequence(size_t epoch) {
  if (epoch2index_seq_.find(epoch) != epoch2index_seq_.end()) { return; }
  std::vector<int64_t> index_seq(dataset_->Size());
  std::iota(index_seq.begin(), index_seq.end(), 0);
  if (dataset_->dataset_proto().shuffle()) {
    std::shuffle(index_seq.begin(), index_seq.end(), gen_);
  }
  CHECK(epoch2index_seq_.emplace(epoch, std::move(index_seq)).second);
  CHECK(epoch2consumed_count_.emplace(epoch, 0).second);
}

void DataSampler::CheckIndexSequenceRanOut(DataSamplerContext* ctx) {
  std::unique_lock<std::mutex> lck(mtx_);
  auto count_it = epoch2consumed_count_.find(ctx->epoch_);
  CHECK(count_it != epoch2consumed_count_.end());
  count_it->second += ctx->count_;
  CHECK_LE(count_it->second, max_count_);
  if (count_it->second == max_count_) {
    auto it = epoch2index_seq_.find(ctx->epoch_);
    CHECK(it != epoch2index_seq_.end());
    epoch2index_seq_.erase(it);
    epoch2consumed_count_.erase(count_it);
  }
}

const std::vector<int64_t>& DataSampler::AcquireGetOrGenEpochIndexSequence(size_t epoch) {
  std::unique_lock<std::mutex> lck(mtx_);
  if (epoch2index_seq_.find(epoch) == epoch2index_seq_.end()) { GenNewEpochIndexSequence(epoch); }
  return epoch2index_seq_.at(epoch);
}

std::vector<int64_t> DataSampler::FetchBatchIndexSequence(DataSamplerContext* ctx,
                                                          size_t batch_size) {
  std::vector<int64_t> batch_index_seq(batch_size);
  size_t i = 0;
  while (i < batch_size) {
    if (ctx->offset_ >= dataset()->Size()) {
      CheckIndexSequenceRanOut(ctx);
      ctx->epoch_ += 1;
      ctx->offset_ %= dataset()->Size();
      ctx->count_ = 0;
    }
    batch_index_seq[i] = AcquireGetOrGenEpochIndexSequence(ctx->epoch_).at(ctx->offset_);
    ctx->offset_ += ctx->num_replicas_;
    ctx->count_ += 1;
    i += 1;
  }
  return batch_index_seq;
}

GroupedDataSampler::GroupedDataSampler(Dataset* dataset)
    : DataSampler(dataset), group_ids_(dataset->Size()) {
  FOR_RANGE(int64_t, i, 0, group_ids_.size()) { group_ids_.at(i) = dataset->GetGroupId(i); }
}

std::vector<int64_t> GroupedDataSampler::FetchBatchIndexSequence(DataSamplerContext* ctx,
                                                                 size_t batch_size) {
  std::vector<int64_t> seq(batch_size);
  bool skip_happened = false;
  int64_t first_group_id = -1;
  size_t offset = ctx->offset_;
  size_t epoch = ctx->epoch_;

  auto MoveOffsetAndRetIndex = [this, ctx, &offset, &epoch]() {
    offset += ctx->num_replicas_;
    if (offset >= dataset()->Size()) {
      epoch += 1;
      offset %= dataset()->Size();
    }
    return AcquireGetOrGenEpochIndexSequence(epoch).at(offset);
  };

  auto FetchIndex = [this, &seq, ctx](size_t i, int64_t index, bool skip_happened) {
    seq.at(i) = index;
    if (!skip_happened) {
      ctx->offset_ += ctx->num_replicas_;
      ctx->count_ += 1;
      if (ctx->offset_ >= dataset()->Size()) {
        CheckIndexSequenceRanOut(ctx);
        ctx->epoch_ += 1;
        ctx->offset_ %= dataset()->Size();
        ctx->count_ = 0;
      }
    }
  };

  int64_t index = AcquireGetOrGenEpochIndexSequence(epoch).at(offset);
  FOR_RANGE(size_t, i, 0, batch_size) {
    if (i == 0) {
      FetchIndex(i, index, skip_happened);
      first_group_id = group_ids_.at(index);
    } else {
      while (first_group_id != group_ids_.at(index)) {
        index = MoveOffsetAndRetIndex();
        skip_happened = true;
      }
      FetchIndex(i, index, skip_happened);
    }
    index = MoveOffsetAndRetIndex();
  }

  return seq;
}

}  // namespace data
}  // namespace oneflow
