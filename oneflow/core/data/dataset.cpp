#include "oneflow/core/data/dataset.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {
namespace data {

Dataset::Dataset(const DatasetProto& dataset_proto) : sampler_(new DataSampler(this)) {
  dataset_proto_ = &dataset_proto;
  data_seq_.clear();
  Init();
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

}  // namespace data
}  // namespace oneflow
