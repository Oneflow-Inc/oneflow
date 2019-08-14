#include "oneflow/core/data/data_sampler.h"
#include "oneflow/core/control/ctrl_client.h"

namespace oneflow {
namespace data {

void DataSampler::GenNewEpochIndexSequence(size_t epoch) {
  std::unique_lock<std::mutex> lck(mtx_);
  if (epoch2index_seq_.find(epoch) != epoch2index_seq_.end()) { return; }
  std::vector<int64_t> index_seq(dataset_->Size());
  std::iota(index_seq.begin(), index_seq.end(), 0);
  if (dataset_->dataset_proto().shuffle()) {
    std::mt19937 gen(dataset_->dataset_proto().random_seed());
    std::shuffle(index_seq.begin(), index_seq.end(), gen);
  }
  CHECK(epoch2index_seq_.emplace(epoch, std::move(index_seq)).second);
}

void DataSampler::DeleteEpochIndexSequence(size_t epoch) {
  auto it = epoch2index_seq_.find(epoch);
  CHECK(it != epoch2index_seq_.end());
  std::unique_lock<std::mutex> lck(mtx_);
  epoch2index_seq_.erase(it);
}

void DataSampler::NotifyIndexSequenceRanOut(size_t epoch, size_t count, size_t iter) {
  CHECK_GE(iter, dataset()->Size());
  auto it = epoch2index_seq_.find(epoch);
  CHECK(it != epoch2index_seq_.end());
  std::string epoch_data_index_count_key = std::to_string(epoch) + "_data_index_count";
  int32_t used_count = Global<CtrlClient>::Get()->IncreaseCount(epoch_data_index_count_key, count);
  CHECK_LE(used_count, it->second.size());
  if (used_count == it->second.size()) { DeleteEpochIndexSequence(epoch); }
}

const std::vector<int64_t>& DataSampler::GetEpochIndexSequence(size_t epoch) const {
  return epoch2index_seq_.at(epoch);
}

std::vector<int64_t> DataSampler::FetchBatchIndexSequence(DataSamplerContext* ctx,
                                                          size_t batch_size) {
  std::vector<int64_t> batch_index_seq(batch_size);
  size_t i = 0;
  while (i < batch_size) {
    if (ctx->iter_ >= dataset()->Size()) {
      NotifyIndexSequenceRanOut(ctx->epoch_, ctx->iter_, ctx->count_);
      ctx->epoch_ += 1;
      ctx->iter_ %= dataset()->Size();
      ctx->count_ = 0;
    }
    batch_index_seq[i] = GetEpochIndexSequence(ctx->epoch_).at(ctx->iter_);
    ctx->iter_ += 1;
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
  bool skip = false;
  size_t i = 0;
  size_t j = ctx->iter_;
  size_t epoch = ctx->epoch_;
  int64_t group_id = group_ids_.at(GetEpochIndexSequence(epoch).at(j));
  while (i < batch_size) {
    if (j >= dataset()->Size()) {
      epoch += 1;
      GenNewEpochIndexSequence(epoch);
      j %= dataset()->Size();
    }
    if (ctx->iter_ >= dataset()->Size()) {
      NotifyIndexSequenceRanOut(ctx->epoch_, ctx->count_, ctx->iter_);
      ctx->epoch_ = epoch;
      ctx->iter_ = j;
      ctx->count_ = 0;
    }
    int64_t index = GetEpochIndexSequence(epoch).at(j);
    int64_t cur_group_id = group_ids_.at(index);
    if (cur_group_id == group_id) {
      seq.at(i) = index;
      ctx->count_ += 1;
      if (!skip) { ctx->iter_ += ctx->num_replicas_; }
      i += 1;
    } else {
      skip = true;
    }
    j += ctx->num_replicas_;
  }
  return seq;
}

}  // namespace data
}  // namespace oneflow
