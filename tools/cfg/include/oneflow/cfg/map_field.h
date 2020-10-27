#ifndef ONEFLOW_CFG_MAP_FIELD_H_
#define ONEFLOW_CFG_MAP_FIELD_H_

#include <map>
#include <string>
#include <memory>
#include <type_traits>
#include "oneflow/cfg/shared_pair_iterator.h"

namespace oneflow {
namespace cfg {

template<typename Key, typename T>
class _ConstMapField_ {
  public:
  static_assert(std::is_nothrow_move_constructible<T>::value, "");
  using key_type = typename std::map<Key, T>::key_type;
  using mapped_type = typename std::map<Key, T>::mapped_type;
  using value_type = typename std::map<Key, T>::value_type;
  using size_type = typename std::map<Key, T>::size_type;
  using difference_type = typename std::map<Key, T>::difference_type;
  using const_reference = typename std::map<Key, T>::const_reference;
  using const_pointer = typename std::map<Key, T>::const_pointer;
  using const_iterator = typename std::map<Key, T>::const_iterator;
  using const_reverse_iterator = typename std::map<Key, T>::const_reverse_iterator;

  using reference = typename std::map<Key, T>::reference;
  using pointer = typename std::map<Key, T>::pointer;
  using iterator = typename std::map<Key, T>::iterator;
  using reverse_iterator = typename std::map<Key, T>::reverse_iterator;

  _ConstMapField_(): data_(std::make_shared<std::map<Key, T>>()) {}
  _ConstMapField_(const std::shared_ptr<std::map<Key, T>>& data): data_(data) {}
  template<typename InputIt>
  _ConstMapField_(InputIt begin, InputIt end): data_(std::make_shared<std::map<Key, T>>(begin, end)) {}
  ~_ConstMapField_() = default;

  iterator begin() noexcept { return data_->begin(); }
  iterator end() noexcept { return data_->end(); }

  const_iterator begin() const noexcept { return data_->begin(); }
  const_iterator cbegin() const noexcept { return data_->cbegin(); }
  const_iterator end() const noexcept { return data_->end(); }
  const_iterator cend() const noexcept { return data_->cend(); }
  const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
  const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend()); }
  const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
  const_reverse_iterator crend() const noexcept { return const_reverse_iterator(cbegin()); }

  bool empty() const { return data_->empty(); }
  size_type size() const { return data_->size(); }
  const T& at(const Key& key) const { return data_->at(key); }
  const T& operator[](const Key& key) const { return (*data_)[key]; }
  const_iterator find(const Key& key) const { return data_->find(key); }
  int count(const Key& key) const { return data_->count(key); }

  bool operator==(const _ConstMapField_& other) const {
    return *__SharedPtr__() == *other.__SharedPtr__();
  }
  bool operator<(const _ConstMapField_& other) const {
    return *__SharedPtr__() < *other.__SharedPtr__();
  }

  const T& Get(const Key& key) const {return at(key);}
  const std::shared_ptr<std::map<Key, T>>& __SharedPtr__() const { return data_; }
  const std::shared_ptr<std::map<Key, T>>& __SharedPtr__() { return data_; }

  std::shared_ptr<_ConstMapField_> __SharedConst__() const { return std::make_shared<_ConstMapField_>(__SharedPtr__());}
  // std::shared_ptr<T> __SharedConst__(const Key& key) const {
  //   return at(key).__SharedConst__();
  // }

  private:
  std::shared_ptr<std::map<Key, T>> data_;

};

template<typename Key, typename T>
class _MapField_: public _ConstMapField_<Key, T>{
 public:
  using reference = typename std::map<Key, T>::reference;
  using pointer = typename std::map<Key, T>::pointer;
  using iterator = typename std::map<Key, T>::iterator;
  using reverse_iterator = typename std::map<Key, T>::reverse_iterator;
  using shared_mut_iterator = _SharedMutPairIterator_<_MapField_, T>;

  using key_type = typename std::map<Key, T>::key_type;
  using mapped_type = typename std::map<Key, T>::mapped_type;
  using value_type = typename std::map<Key, T>::value_type;
  using size_type = typename std::map<Key, T>::size_type;
  using difference_type = typename std::map<Key, T>::difference_type;
  using const_reference = typename std::map<Key, T>::const_reference;
  using const_pointer = typename std::map<Key, T>::const_pointer;
  using const_iterator = typename std::map<Key, T>::const_iterator;
  using const_reverse_iterator = typename std::map<Key, T>::const_reverse_iterator;

  using _ConstMapField_<Key, T>::begin;
  using _ConstMapField_<Key, T>::end;
  using _ConstMapField_<Key, T>::at;

  _MapField_(): data_(std::make_shared<std::map<Key, T>>()) {}
  _MapField_(const std::shared_ptr<std::map<Key, T>>& data): data_(data) {}
  _MapField_(const _MapField_& other): data_(std::make_shared<std::map<Key, T>>()) {
    CopyFrom(other);
  }

  _MapField_(const _ConstMapField_<Key, T>& other): data_(std::make_shared<std::map<Key, T>>()) {
    CopyFrom(other);
  }

  _MapField_(_MapField_&&) = default;
  template<typename InputIt>
  _MapField_(InputIt begin, InputIt end): data_(std::make_shared<std::map<Key, T>>(begin, end)) {}
  ~_MapField_() = default;

  //iterator begin() noexcept { return data_->begin(); }
  //iterator end() noexcept { return data_->end(); }
  reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
  reverse_iterator rend() noexcept { return reverse_iterator(begin()); }

  T& at(const Key& key) { return data_->at(key); }
  T& operator[](const Key& key) { return (*data_)[key]; }

  iterator find(const Key& key) { return data_->find(key); }
  size_type erase(const Key& key) { return data_->erase(key); }
  iterator erase(const_iterator pos) { return data_->erase(pos); }
  iterator erase(const_iterator first, const_iterator last) { return data_->erase(first, last); }

  std::pair<iterator, bool> insert(const value_type& value) { return data_->insert(value); }
  template<class InputIt>
  void insert(InputIt first, InputIt last) { return data_->insert(first, last); }

  void Clear() { data_->clear(); }
  void CopyFrom(const _MapField_& other) {
    *data_ = *other.data_;
  }

  void CopyFrom(const _ConstMapField_<Key, T>& other) {
    *data_ = *other.__SharedPtr__();
  }
  _MapField_& operator=(const _MapField_& other) {
    CopyFrom(other);
    return *this;
  }

  bool operator==(const _MapField_& other) const {
    return *__SharedPtr__() == *other.__SharedPtr__();
  }
  bool operator<(const _MapField_& other) const {
    return *__SharedPtr__() < *other.__SharedPtr__();
  }

  void Set(const Key& key, const T& value){
    (*this)[key] = value;
  }

  const std::shared_ptr<std::map<Key, T>>& __SharedPtr__() const { return data_; }
  const std::shared_ptr<std::map<Key, T>>& __SharedPtr__() { return data_; }

  std::shared_ptr<_MapField_> __SharedMutable__() {
    return std::make_shared<_MapField_>(__SharedPtr__());
  }
  // std::shared_ptr<T> __SharedMutable__(const Key& key) {
  //   return (*this)[key].__SharedMutable__();
  // }

  shared_mut_iterator shared_mut_begin() { return begin(); }
  shared_mut_iterator shared_mut_end() { return end(); }

 private:
  std::shared_ptr<std::map<Key, T>> data_;
};

}
}

#endif  // ONEFLOW_CFG_MAP_FIELD_H_
