#ifndef ONEFLOW_CFG_REPEATED_FIELD_H_
#define ONEFLOW_CFG_REPEATED_FIELD_H_

#include <vector>
#include <string>
#include <memory>
#include <type_traits>

namespace oneflow {
namespace cfg {

template<typename T>
class _ConstRepeatedField_ {
  public:

  static_assert(std::is_nothrow_move_constructible<T>::value, "");
  using value_type = typename std::vector<T>::value_type;
  using size_type = typename std::vector<T>::size_type;
  using difference_type = typename std::vector<T>::difference_type;
  using const_reference = typename std::vector<T>::const_reference;
  using const_pointer = typename std::vector<T>::const_pointer;
  using const_iterator = typename std::vector<T>::const_iterator;
  using const_reverse_iterator = typename std::vector<T>::const_reverse_iterator;

  _ConstRepeatedField_():  data_(std::make_shared<std::vector<T>>()) {}
  _ConstRepeatedField_(const std::shared_ptr<std::vector<T>>& data): data_(data) {}
  template<typename InputIt>
  _ConstRepeatedField_(InputIt begin, InputIt end): data_(std::make_shared<std::vector<T>>(begin, end)) {}
  ~_ConstRepeatedField_() = default;

  const_iterator begin() const noexcept { return data_->begin(); }
  const_iterator cbegin() const noexcept { return data_->cbegin(); }

  const_iterator end() const noexcept { return data_->end(); }
  const_iterator cend() const noexcept { return data_->end(); }

  const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
  const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend()); }

  const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
  const_reverse_iterator crend() const noexcept { return const_reverse_iterator(cbegin()); }

  bool empty() const { return data_->empty(); }
  size_type size() const { return data_->size(); }
  const_reference at(size_type pos) const { return data_->at(pos); }
  const_reference operator[](size_type pos) const { return data_[pos]; }
  const_reference Get(size_type pos) const { return data_->at(pos); }

  const std::shared_ptr<std::vector<T>>& __SharedPtr__() const { return data_; }
  const std::shared_ptr<std::vector<T>>& __SharedPtr__() { return data_; }

  std::shared_ptr<_ConstRepeatedField_> __SharedConst__() const {
    return std::make_shared<_ConstRepeatedField_>(__SharedPtr__());
  }

  bool operator==(const _ConstRepeatedField_& other) const {return *__SharedPtr__() == *other.__SharedPtr__();}
  bool operator<(const _ConstRepeatedField_& other) const {return *__SharedPtr__() < *other.__SharedPtr__();}


  private:
  std::shared_ptr<std::vector<T>> data_;

};

template<typename T>
class _RepeatedField_: public _ConstRepeatedField_<T>{
 public:
  static_assert(std::is_nothrow_move_constructible<T>::value, "");
  using reference = typename std::vector<T>::reference;
  using pointer = typename std::vector<T>::pointer;
  using iterator = typename std::vector<T>::iterator;
  using reverse_iterator = typename std::vector<T>::reverse_iterator;

  using value_type = typename std::vector<T>::value_type;
  using size_type = typename std::vector<T>::size_type;
  using difference_type = typename std::vector<T>::difference_type;
  using const_reference = typename std::vector<T>::const_reference;
  using const_pointer = typename std::vector<T>::const_pointer;
  using const_iterator = typename std::vector<T>::const_iterator;
  using const_reverse_iterator = typename std::vector<T>::const_reverse_iterator;

  using _ConstRepeatedField_<T>::begin;
  using _ConstRepeatedField_<T>::end;
  using _ConstRepeatedField_<T>::at;

  _RepeatedField_(): data_(std::make_shared<std::vector<T>>()) {}
  _RepeatedField_(const std::shared_ptr<std::vector<T>>& data): data_(data) {}
  _RepeatedField_(const _RepeatedField_& other): data_(std::make_shared<std::vector<T>>()) {
    CopyFrom(other);
  }

  _RepeatedField_(const _ConstRepeatedField_<T>& other): data_(std::make_shared<std::vector<T>>()) {
    CopyFrom(other);
  }

  _RepeatedField_(_RepeatedField_&&) = default;
  template<typename InputIt>
  _RepeatedField_(InputIt begin, InputIt end): data_(std::make_shared<std::vector<T>>(begin, end)) {}
  ~_RepeatedField_() = default;

  iterator begin() noexcept { return data_->begin(); }
  iterator end() noexcept { return data_->end(); }

  reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
  reverse_iterator rend() noexcept { return reverse_iterator(begin()); }

  reference at(size_type pos) { return data_->at(pos); }
  reference operator[](size_type pos) { return data_[pos]; }

  const std::shared_ptr<std::vector<T>>& __SharedPtr__() const { return data_; }
  const std::shared_ptr<std::vector<T>>& __SharedPtr__() { return data_; }


  std::shared_ptr<_RepeatedField_> __SharedMutable__() {
    return std::make_shared<_RepeatedField_>(__SharedPtr__());
  }

  std::shared_ptr<T> __SharedMutable__(std::size_t index) {
    return Mutable(index)->__SharedMutable__();
  }

  std::shared_ptr<T> __SharedAdd__() {
    return Add()->__SharedMutable__();
  }

  pointer Mutable(size_type pos) { return &data_->at(pos); }

  bool operator==(const _RepeatedField_& other) const {
    return *__SharedPtr__() == *other.__SharedPtr__();
  }
  bool operator<(const _RepeatedField_& other) const {
    return *__SharedPtr__() < *other.__SharedPtr__();
  }

  void Clear() { data_->clear(); }
  void CopyFrom(const _RepeatedField_& other) {
    if (std::is_scalar<T>::value || std::is_same<std::string, T>::value) {
      *data_ = *other.data_;
    } else {
      data_->clear();
      for (const auto& elem : other) { *Add() = elem; }
    }
  }

  void CopyFrom(const _ConstRepeatedField_<T>& other) {
    CopyFrom(other);
  }

  _RepeatedField_& operator=(const _RepeatedField_& other) {
    CopyFrom(other);
    return *this;
  }

  void Set(size_type pos, const T& elem) {
    data_->at(pos) = elem;
  }
  void Add(const T& elem) {
    data_->push_back(std::move(elem));
  }
  pointer Add() {
    data_->push_back(T());
    return &data_->at(data_->size() - 1);
  }

 private:
  std::shared_ptr<std::vector<T>> data_;
};

}
}

#endif  // ONEFLOW_CFG_REPEATED_FIELD_H_
