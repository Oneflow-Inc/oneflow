#ifndef ONEFLOW_CORE_VM_STORAGE_H_
#define ONEFLOW_CORE_VM_STORAGE_H_

#include <mutex>
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

template<typename T>
class Storage final {
 public:
  Storage(const Storage&) = delete;
  Storage(Storage&&) = delete;

  Storage() = default;
  ~Storage() = default;

  bool Has(int64_t logical_object_id) const {
    std::unique_lock<std::mutex> lock(mutex_);
    return logical_object_id2data_.find(logical_object_id) != logical_object_id2data_.end();
  }

  const T& Get(int64_t logical_object_id) const { return *GetPtr(logical_object_id); }

  const std::shared_ptr<T>& GetPtr(int64_t logical_object_id) const {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto& iter = logical_object_id2data_.find(logical_object_id);
    CHECK(iter != logical_object_id2data_.end());
    return iter->second;
  }

  void Add(int64_t logical_object_id, const std::shared_ptr<T>& data) {
    CHECK_GT(logical_object_id, 0);
    std::unique_lock<std::mutex> lock(mutex_);
    CHECK(logical_object_id2data_.emplace(logical_object_id, data).second);
  }
  void Clear(int64_t logical_object_id) {
    std::unique_lock<std::mutex> lock(mutex_);
    logical_object_id2data_.erase(logical_object_id);
  }
  void ClearAll() {
    std::unique_lock<std::mutex> lock(mutex_);
    logical_object_id2data_.clear();
  }

 private:
  mutable std::mutex mutex_;
  HashMap<int64_t, std::shared_ptr<T>> logical_object_id2data_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_STORAGE_H_
