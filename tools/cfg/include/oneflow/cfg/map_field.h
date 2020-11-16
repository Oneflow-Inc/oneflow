#ifndef ONEFLOW_CFG_MAP_FIELD_H_
#define ONEFLOW_CFG_MAP_FIELD_H_

#include <map>
#include <string>
#include <memory>
#include <type_traits>

namespace oneflow {
namespace cfg {

template<typename Key, typename T>
class _MapField_ {
 public:
  using key_type = typename std::map<Key, T>::key_type;
  using mapped_type = typename std::map<Key, T>::mapped_type;
  using value_type = typename std::map<Key, T>::value_type;
  using size_type = typename std::map<Key, T>::size_type;
  using difference_type = typename std::map<Key, T>::difference_type;
  using reference = typename std::map<Key, T>::reference;
  using const_reference = typename std::map<Key, T>::const_reference;
  using pointer = typename std::map<Key, T>::pointer;
  using const_pointer = typename std::map<Key, T>::const_pointer;
  using iterator = typename std::map<Key, T>::iterator;
  using const_iterator = typename std::map<Key, T>::const_iterator;
  using reverse_iterator = typename std::map<Key, T>::reverse_iterator;
  using const_reverse_iterator = typename std::map<Key, T>::const_reverse_iterator;

  _MapField_(): data_(std::make_shared<std::map<Key, T>>()) {}
  _MapField_(const std::shared_ptr<std::map<Key, T>>& data): data_(data) {}
  _MapField_(const _MapField_& other): data_(std::make_shared<std::map<Key, T>>()) {
    CopyFrom(other);
  }
  _MapField_(_MapField_&&) = default;
  template<typename InputIt>
  _MapField_(InputIt begin, InputIt end): data_(std::make_shared<std::map<Key, T>>(begin, end)) {}
  ~_MapField_() = default;

  iterator begin() noexcept { return data_->begin(); }
  const_iterator begin() const noexcept { return data_->begin(); }
  const_iterator cbegin() const noexcept { return data_->cbegin(); }

  iterator end() noexcept { return data_->end(); }
  const_iterator end() const noexcept { return data_->end(); }
  const_iterator cend() const noexcept { return data_->cend(); }

  reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
  const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
  const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend()); }

  reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
  const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
  const_reverse_iterator crend() const noexcept { return const_reverse_iterator(cbegin()); }

  bool empty() const { return data_->empty(); }
  size_type size() const { return data_->size(); }
  T& at(const Key& key) { return data_->at(key); }
  const T& at(const Key& key) const { return data_->at(key); }
  T& operator[](const Key& key) { return (*data_)[key]; }
  const T& operator[](const Key& key) const { return (*data_)[key]; }

  const_iterator find(const Key& key) const { return data_->find(key); }
  iterator find(const Key& key) { return data_->find(key); }
  int count(const Key& key) const { return data_->count(key); }

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

  _MapField_& operator=(const _MapField_& other) {
    CopyFrom(other);
    return *this;
  }

  const std::shared_ptr<std::map<Key, T>>& __SharedPtr__() const { return data_; }

  const std::shared_ptr<std::map<Key, T>>& __SharedPtr__() { return data_; }

 private:
  std::shared_ptr<std::map<Key, T>> data_;
};

}
}

#endif  // ONEFLOW_CFG_MAP_FIELD_H_
