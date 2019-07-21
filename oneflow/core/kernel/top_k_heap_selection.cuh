#ifndef ONEFLOW_CORE_KERNEL_TOP_K_HEAP_SELECTION_CUH_
#define ONEFLOW_CORE_KERNEL_TOP_K_HEAP_SELECTION_CUH_

namespace oneflow {

template<typename T>
class Entry {
 public:
  __device__ Entry(int32_t index, T value) : index_(index), value_(value) {}
  __device__ const int32_t GetIndex() const { return index_; }
  __device__ const T GetValue() const { return value_; }
  __device__ void SetIndex(int32_t index) { index_ = index; }
  __device__ void SetValue(T value) { value_ = value; }
  __device__ Entry<T>& operator=(const Entry<T>& entry) {
    this->index_ = entry.GetIndex();
    this->value_ = entry.GetValue();
  }
  __device__ bool operator==(const Entry& entry) {
    return this->GetValue() == entry.GetValue() && this->GetIndex() == entry.GetIndex();
  }
  __device__ bool operator!=(const Entry& entry) {
    return this->GetValue() != entry.GetValue() || this->GetIndex() != entry.GetIndex();
  }
  __device__ bool operator<(const Entry& entry) {
    if (this->GetValue() == entry.GetValue()) { return this->GetIndex() > entry.GetIndex(); }
    return this->GetValue() < entry.GetValue();
  }
  __device__ bool operator<=(const Entry& entry) {
    if (this->GetValue() == entry.GetValue()) { return this->GetIndex() >= entry.GetIndex(); }
    return this->GetValue() <= entry.GetValue();
  }
  __device__ bool operator>(const Entry& entry) {
    if (this->GetValue() == entry.GetValue()) { return this->GetIndex() < entry.GetIndex(); }
    return this->GetValue() > entry.GetValue();
  }
  __device__ bool operator>=(const Entry& entry) {
    if (this->GetValue() == entry.GetValue()) { return this->GetIndex() <= entry.GetIndex(); }
    return this->GetValue() >= entry.GetValue();
  }

 private:
  int32_t index_;
  T value_;
};

template<typename T>
struct EntryGTComp {
  __device__ bool operator()(const T& x, const T& y) {
    return x.GetValue() > y.GetValue()
           || (x.GetValue() == y.GetValue() && (x.GetIndex() < y.GetIndex()));
  }
};

template<typename T>
class Heap {
 public:
  __device__ Heap(Entry<T>* data, const int32_t heap_size, const int32_t init_index,
                  const T init_value)
      : data_(data), heap_size_(heap_size) {
    for (int32_t i = 0; i < heap_size; ++i) {
      data_[i].SetIndex(init_index);
      data_[i].SetValue(init_value);
    }
  }
  __device__ Entry<T>& operator[](const int32_t index) const { return data_[index]; }
  __device__ const int32_t Left(const int32_t index) const { return 2 * index + 1; }
  __device__ const int32_t Right(const int32_t index) const { return 2 * index + 2; }
  __device__ void Swap(const int32_t i, const int32_t j) {
    auto tmp = data_[j];
    data_[j] = data_[i];
    data_[i] = tmp;
  }
  __device__ void Heapify(int32_t index) {
    while (true) {
      const int32_t left = Left(index);
      const int32_t right = Right(index);
      int32_t min = index;
      if (left < heap_size_ && data_[left] < data_[index]) { min = left; }
      if (right < heap_size_ && data_[right] < data_[index]) { min = right; }
      if (min == index) { break; }
      Swap(min, index);
      index = min;
    }
  }
  __device__ void ReplaceRoot(Entry<T> entry) {
    data_[0] = entry;
    Heapify(0);
  }

 private:
  Entry<T>* data_;
  int32_t heap_size_;
};

}  // namespace oneflow

#endif
