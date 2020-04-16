#ifndef ONEFLOW_CUSTOMIZED_DATA_DATASET_H_
#define ONEFLOW_CUSTOMIZED_DATA_DATASET_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/tensor_buffer.h"

namespace oneflow {

template<typename LoadTarget>
class DataSet {
 public:
  using LoadTargetPtr = std::shared_ptr<LoadTarget>;
  using BatchLoadTargetPtr = std::vector<LoadTargetPtr>;
  DataSet() = default;
  ~DataSet() = default;

  virtual BatchLoadTargetPtr LoadBatch(int64_t batch_size) { UNIMPLEMENTED(); }
  virtual void Next(LoadTarget& tensor) = 0;
  virtual void At(int64_t idx, LoadTarget& tensor) { UNIMPLEMENTED(); }

  virtual int64_t Size() {
    UNIMPLEMENTED();
    return -1;
  }

  virtual bool EnableRandomAccess() { return false; }
  virtual bool EnableGetSize() { return false; }
};

static constexpr int kOneflowDataSetSeed = 524287;

template<typename LoadTarget>
class EmptyTensorManager {
 public:
  using LoadTargetUniquePtr = std::unique_ptr<LoadTarget>;
  EmptyTensorManager(int64_t total_empty_size, int64_t tensor_init_bytes)
      : total_empty_size_(total_empty_size), tensor_init_bytes_(tensor_init_bytes) {
    for (int i = 0; i < total_empty_size; ++i) {
      auto tensor_ptr = LoadTargetUniquePtr(new LoadTarget());
      PrepareEmpty(*tensor_ptr);
      empty_tensors_.push_back(std::move(tensor_ptr));
    }
  }

  void Recycle(LoadTarget* tensor_ptr) {
    LoadTargetUniquePtr recycle_ptr(tensor_ptr);
    std::lock_guard<std::mutex> lock(empty_tensors_mutex_);
    empty_tensors_.push_back(std::move(recycle_ptr));
  }

  LoadTargetUniquePtr Get() {
    LoadTargetUniquePtr ret;
    std::lock_guard<std::mutex> lock(empty_tensors_mutex_);
    CHECK_GT(empty_tensors_.size(), 0);
    ret = std::move(empty_tensors_.back());
    empty_tensors_.pop_back();
    return ret;
  }

 protected:
  virtual void PrepareEmpty(LoadTarget& tensor) { PrepareEmptyTensor(tensor); }

 private:
  template<typename T>
  std::enable_if_t<std::is_same<T, TensorBuffer>::value> PrepareEmptyTensor(T& tensor) {
    tensor.reset();
    // Initialize tensors to a set size to limit expensive reallocations
    tensor.Resize({tensor_init_bytes_}, DataType::kUInt8);
  }

  template<typename T>
  std::enable_if_t<!std::is_same<T, TensorBuffer>::value> PrepareEmptyTensor(T&) {
    constexpr bool T_is_TensorBuffer = std::is_same<T, TensorBuffer>::value;
    CHECK(T_is_TensorBuffer)
        << "Please overload PrepareEmpty for custom LoadTarget type other than TensorBuffer";
  }

  const int total_empty_size_;
  const int tensor_init_bytes_;

  std::mutex empty_tensors_mutex_;
  std::vector<LoadTargetUniquePtr> empty_tensors_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_DATASET_H_
