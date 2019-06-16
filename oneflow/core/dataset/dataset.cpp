#include "oneflow/core/dataset/dataset.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void Dataset::Init(const DatasetProto& dataset_proto) {
  dataset_proto_ = &dataset_proto;
  data_seq_.clear();
  // status_.reset(new DatasetStatus);
  VirtualInit();
}

void Dataset::GenDataSequence(int64_t total_data_num) {
  int64_t epoch = static_cast<int64_t>(std::ceil(total_data_num * 1.0f / this->Size()));
  FOR_RANGE(int64_t, i, 0, epoch) {
    std::vector<int64_t> epoch_seq(this->Size());
    std::iota(epoch_seq.begin(), epoch_seq.end(), 0);
    if (dataset_proto_->shuffle()) {
      std::mt19937 gen(dataset_proto_->random_seed());
      std::shuffle(epoch_seq.begin(), epoch_seq.end(), gen);
    }
    data_seq_.insert(data_seq_.end(), epoch_seq.begin(), epoch_seq.end());
  }
  data_seq_.resize(total_data_num);
}

std::vector<int64_t> Dataset::GetPartDataSequence(int64_t part_id, int64_t part_num) const {
  CHECK_LE(part_num, data_seq_.size());
  BalancedSplitter bs(data_seq_.size(), part_num);
  Range range = bs.At(part_id);
  return std::vector<int64_t>(range.begin(), range.end());
}

// int64_t Dataset::GenNewEpochDataSequence(int64_t epoch) {
//   int64_t step = status_->ForwardEpoch(epoch);
//   status_->GenDataSequence(step, this->Size(), dataset_proto_.shuffle(),
//   dataset_proto_.random_seed());
// }

// int64_t DatasetStatus::ForwardEpoch(int64_t epoch) {
//   std::unique_lock<std::mutex> lck(mtx_);
//   int64_t step = epoch - epoch_;
// }

// void DatasetStatus::GenDataSequence(int64_t count, int64_t size, bool shuffle, int64_t
// random_seed) {
//   if (count <= 0) { return; }

// }

}  // namespace oneflow
