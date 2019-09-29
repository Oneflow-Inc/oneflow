#ifndef ONEFLOW_CORE_DATA_DATA_SAMPLER_H_
#define ONEFLOW_CORE_DATA_DATA_SAMPLER_H_

namespace oneflow {
namespace data {

struct DataSamplerContext final {
  size_t num_replicas_;
  size_t rank_;
  size_t epoch_;
  size_t iter_;
  size_t count_;
};

class Dataset;

class DataSampler {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataSampler);
  DataSampler() = delete;
  virtual ~DataSampler() = default;
  DataSampler(Dataset* dataset) : dataset_(dataset) {}

  virtual std::vector<int64_t> FetchBatchIndexSequence(DataSamplerContext* ctx, size_t batch_size);
  void GenNewEpochIndexSequence(size_t epoch);
  void DeleteEpochIndexSequence(size_t epoch);
  void NotifyIndexSequenceRanOut(size_t epoch, size_t count, size_t iter);

  const std::vector<int64_t>& GetEpochIndexSequence(size_t epoch) const;
  const std::vector<int64_t>& TryGetEpochIndexSequence(size_t epoch);
  Dataset* dataset() { return dataset_; }

 private:
  Dataset* dataset_;
  HashMap<size_t, std::vector<int64_t>> epoch2index_seq_;

  std::mutex mtx_;
};

class GroupedDataSampler : public DataSampler {
 public:
  GroupedDataSampler(Dataset* dataset);
  virtual std::vector<int64_t> FetchBatchIndexSequence(DataSamplerContext* ctx,
                                                       size_t batch_size) override;

 private:
  std::vector<int64_t> group_ids_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_CORE_DATA_DATA_SAMPLER_H_
