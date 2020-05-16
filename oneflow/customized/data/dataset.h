#ifndef ONEFLOW_CUSTOMIZED_DATA_DATASET_H_
#define ONEFLOW_CUSTOMIZED_DATA_DATASET_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/tensor_buffer.h"

namespace oneflow {

template<typename LoadTarget>
class Dataset {
 public:
  using LoadTargetPtr = std::shared_ptr<LoadTarget>;
  using LoadTargetPtrList = std::vector<LoadTargetPtr>;
  Dataset() = default;
  virtual ~Dataset() = default;

  virtual LoadTargetPtrList Next() = 0;
};

static constexpr int kOneflowDatasetSeed = 524287;

template<typename LoadTarget>
void PrepareEmptyTensor(LoadTarget& tensor, int32_t tensor_init_bytes);

template<typename LoadTarget>
class EmptyTensorManager final {
 public:
  using LoadTargetUniquePtr = std::unique_ptr<LoadTarget>;
  using LoadTargetSharedPtr = std::shared_ptr<LoadTarget>;
  EmptyTensorManager(int64_t total_empty_size, int32_t tensor_init_bytes)
      : tensor_init_bytes_(tensor_init_bytes), total_tensor_count_(0) {
    for (int i = 0; i < total_empty_size; ++i) {
      auto tensor_ptr = LoadTargetUniquePtr(new LoadTarget());
      PrepareEmptyTensor<LoadTarget>(*tensor_ptr, tensor_init_bytes_);
      empty_tensors_.push_back(std::move(tensor_ptr));
    }
    total_tensor_count_ += total_empty_size;
  }
  ~EmptyTensorManager() = default;

  void Recycle(LoadTarget* tensor_ptr) {
    LoadTargetUniquePtr recycle_ptr(tensor_ptr);
    std::lock_guard<std::mutex> lock(empty_tensors_mutex_);
    empty_tensors_.push_back(std::move(recycle_ptr));
  }

  LoadTargetSharedPtr Get() {
    {
      std::lock_guard<std::mutex> lock(empty_tensors_mutex_);
      if (empty_tensors_.size() > 0) {
        LoadTargetSharedPtr ret(empty_tensors_.back().release(),
                                [this](LoadTarget* sample) { Recycle(sample); });
        empty_tensors_.pop_back();
        return ret;
      }
    }
    auto tensor_ptr =
        LoadTargetSharedPtr(new LoadTarget(), [this](LoadTarget* sample) { Recycle(sample); });
    PrepareEmptyTensor<LoadTarget>(*tensor_ptr, tensor_init_bytes_);
    total_tensor_count_++;
    LOG(INFO) << "empty tensor is NOT enough , so we allocate one. The total tensor count is "
              << total_tensor_count_;
    return tensor_ptr;
  }

 private:
  const int32_t tensor_init_bytes_;

  int64_t total_tensor_count_;

  std::mutex empty_tensors_mutex_;
  std::vector<LoadTargetUniquePtr> empty_tensors_;
};

template<>
void PrepareEmptyTensor(TensorBuffer& tensor, int32_t tensor_init_bytes);

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_DATASET_H_
