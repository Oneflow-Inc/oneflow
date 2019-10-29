#ifndef ONEFLOW_CORE_COMMON_FIXED_VECTOR_H_
#define ONEFLOW_CORE_COMMON_FIXED_VECTOR_H_

#include <vector>
#include <initializer_list>

namespace oneflow {

template<typename T, long long kMaxSize>
class fixed_vector final {
 public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = T*;
  using const_pointer = const T*;
  using iterator = typename std::vector<T>::iterator;
  using const_iterator = typename std::vector<T>::const_iterator;
  using reverse_iterator = typename std::vector<T>::reverse_iterator;
  using const_reverse_iterator = typename std::vector<T>::const_reverse_iterator;

  fixed_vector() = default;
  explicit fixed_vector(size_t size) : vec_(size) {}
  explicit fixed_vector(size_t size, const T& val) : vec_(size, val) {}
  template<class InputIt>
  fixed_vector(InputIt first, InputIt last) : vec_(first, last) {}
  fixed_vector(const fixed_vector&) = default;
  fixed_vector(fixed_vector&&) = default;
  fixed_vector(std::initializer_list<T> vec) : vec_(vec) {}
  ~fixed_vector() = default;

  fixed_vector& operator=(const fixed_vector& other) {
    vec_ = other.vec_;
    return *this;
  }
  fixed_vector& operator=(fixed_vector&& other) noexcept {
    vec_ = std::move(other.vec_);
    return *this;
  }
  fixed_vector& operator=(std::initializer_list<T> ilist) {
    vec_ = ilist;
    return *this;
  }

  void assign(size_type count, const T& value) { return vec_.assign(count, value); }
  template<class InputIt>
  void assign(InputIt first, InputIt last) {
    return vec_.assign(first, last);
  }
  void assign(std::initializer_list<T> ilist) { return vec_.assign(ilist); }

  reference at(size_type pos) { return vec_.at(pos); }
  const_reference at(size_type pos) const { return vec_.at(pos); }

  reference operator[](size_type pos) { return vec_[pos]; }
  const_reference operator[](size_type pos) const { return vec_[pos]; }

  reference front() { return vec_.front(); }
  const_reference front() const { return vec_.front(); }

  reference back() { return vec_.back(); }
  const_reference back() const { return vec_.back(); }

  T* data() noexcept { return vec_.data(); }
  const T* data() const noexcept { return vec_.data(); }

  iterator begin() noexcept { return vec_.begin(); }
  const_iterator begin() const noexcept { return vec_.cbegin(); }
  const_iterator cbegin() const noexcept { return vec_.cbegin(); }

  iterator end() noexcept { return vec_.end(); }
  const_iterator end() const noexcept { return vec_.cend(); }
  const_iterator cend() const noexcept { return vec_.cend(); }

  reverse_iterator rbegin() noexcept { return vec_.rbegin(); }
  const_reverse_iterator rbegin() const noexcept { return vec_.crbegin(); }
  const_reverse_iterator crbegin() const noexcept { return vec_.crbegin(); }

  reverse_iterator rend() noexcept { return vec_.rend(); }
  const_reverse_iterator rend() const noexcept { return vec_.crend(); }
  const_reverse_iterator crend() const noexcept { return vec_.crend(); }

  bool empty() const noexcept { return vec_.empty(); }

  size_type size() const noexcept { return vec_.size(); }

  size_type max_size() const noexcept { return vec_.max_size(); }

  size_type capacity() const noexcept { return vec_.capacity(); }

  void clear() noexcept { return vec_.clear(); }

  iterator insert(iterator pos, const T& value) { return vec_.insert(pos, value); }
  iterator insert(iterator pos, T&& value) { return vec_.insert(pos, std::move(value)); }
  iterator insert(iterator pos, size_type count, const T& value) {
    return vec_.insert(pos, count, value);
  }
  template<class InputIt>
  void insert(iterator pos, InputIt first, InputIt last) {
    return vec_.insert(pos, first, last);
  }
  iterator insert(iterator pos, std::initializer_list<T> ilist) { return vec_.insert(pos, ilist); }

  template<class... Args>
  iterator emplace(iterator pos, Args&&... args) {
    return vec_.emplace(pos, std::forward<Args>(args)...);
  }

  iterator erase(iterator pos) { return vec_.erase(pos); }
  iterator erase(iterator first, iterator last) { return vec_.erase(first, last); }

  void push_back(const T& value) { return vec_.push_back(value); }
  void push_back(T&& value) { return vec_.push_back(std::move(value)); }

  template<class... Args>
  void emplace_back(Args&&... args) {
    return vec_.emplace_back(std::forward<Args>(args)...);
  }

  void pop_back() { return vec_.pop_back(); }

  void resize(size_type count) { return vec_.resize(count); }
  void resize(size_type count, const value_type& value) { return vec_.resize(count, value); }

  void swap(fixed_vector& other) noexcept { return vec_.swap(other.vec_); }

  std::vector<T> vec_;
};

template<class T, long long kMaxSize>
void swap(fixed_vector<T, kMaxSize>& lhs, fixed_vector<T, kMaxSize>& rhs) {
  return std::swap(lhs.vec_, rhs.vec_);
}

template<class T, long long kMaxSize>
bool operator==(const fixed_vector<T, kMaxSize>& lhs, const fixed_vector<T, kMaxSize>& rhs) {
  return lhs.vec_ == rhs.vec_;
}

template<class T, long long kMaxSize>
bool operator!=(const fixed_vector<T, kMaxSize>& lhs, const fixed_vector<T, kMaxSize>& rhs) {
  return lhs.vec_ != rhs.vec_;
}

template<class T, long long kMaxSize>
bool operator>=(const fixed_vector<T, kMaxSize>& lhs, const fixed_vector<T, kMaxSize>& rhs) {
  return lhs.vec_ >= rhs.vec_;
}

template<class T, long long kMaxSize>
bool operator>(const fixed_vector<T, kMaxSize>& lhs, const fixed_vector<T, kMaxSize>& rhs) {
  return lhs.vec_ > rhs.vec_;
}

template<class T, long long kMaxSize>
bool operator<=(const fixed_vector<T, kMaxSize>& lhs, const fixed_vector<T, kMaxSize>& rhs) {
  return lhs.vec_ <= rhs.vec_;
}

template<class T, long long kMaxSize>
bool operator<(const fixed_vector<T, kMaxSize>& lhs, const fixed_vector<T, kMaxSize>& rhs) {
  return lhs.vec_ < rhs.vec_;
}

#define SHAPE_MAX_AXIS_SIZE 20

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_FIXED_VECTOR_H_
