#include "oneflow/customized/data/empty_tensor_manager.h"
#include "oneflow/customized/data/coco_dataset.h"
#include "oneflow/core/common/tensor_buffer.h"

namespace oneflow {
namespace data {

template<typename LoadTarget>
EmptyTensorManager<LoadTarget>::EmptyTensorManager(int64_t total_empty_size,
                                                   int64_t tensor_init_bytes)
    : tensor_init_bytes_(tensor_init_bytes), total_tensor_count_(0) {
  for (int i = 0; i < total_empty_size; ++i) {
    auto tensor_ptr = LoadTargetUniquePtr(new LoadTarget());
    PrepareEmptyTensor<LoadTarget>(*tensor_ptr, tensor_init_bytes_);
    empty_tensors_.push_back(std::move(tensor_ptr));
  }
  total_tensor_count_ += total_empty_size;
}

template<typename LoadTarget>
void EmptyTensorManager<LoadTarget>::Recycle(LoadTarget* tensor_ptr) {
  LoadTargetUniquePtr recycle_ptr(tensor_ptr);
  std::lock_guard<std::mutex> lock(empty_tensors_mutex_);
  empty_tensors_.push_back(std::move(recycle_ptr));
}

template<typename LoadTarget>
std::shared_ptr<LoadTarget> EmptyTensorManager<LoadTarget>::Get() {
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

template<>
void PrepareEmptyTensor(TensorBuffer& tensor, int64_t tensor_init_bytes) {
  tensor.reset();
  // Initialize tensors to a set size to limit expensive reallocations
  tensor.Resize({tensor_init_bytes}, DataType::kChar);
}

template<>
void PrepareEmptyTensor(COCOImage& image, int64_t tensor_init_bytes) {
  image.data.Resize({tensor_init_bytes}, DataType::kChar);
}

template class EmptyTensorManager<TensorBuffer>;
template class EmptyTensorManager<COCOImage>;

}  // namespace data
}  // namespace oneflow
