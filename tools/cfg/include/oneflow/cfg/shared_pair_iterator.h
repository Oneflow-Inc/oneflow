#ifndef CFG_ONEFLOW_CFG_SHARED_PAIR_ITERATOR_H_
#define CFG_ONEFLOW_CFG_SHARED_PAIR_ITERATOR_H_

#include <iterator>
#include <memory>

namespace oneflow {
namespace cfg {

template<typename MapT, typename T, typename SharedUtil>
class _SharedPairIterator_ {
 public:
  using DataIter = typename MapT::iterator;
  using key_type = typename MapT::key_type;
  using mapped_type = std::shared_ptr<T>;
  using value_type = std::pair<const key_type, mapped_type>;
  using iterator_category = typename DataIter::iterator_category;
  using self_type = _SharedPairIterator_;
  using pointer = std::unique_ptr<value_type>;
  using reference = value_type;

  _SharedPairIterator_(DataIter data_iter)
      : data_iter_(data_iter) {}

  // const methods

  bool operator==(const _SharedPairIterator_& rhs) const {
    return data_iter_ == rhs.data_iter_;
  }

  bool operator!=(const _SharedPairIterator_& rhs) const { return !(*this == rhs); }

  const pointer operator->() const {
    auto* raw_ptr = new value_type(data_iter_->first, SharedUtil::Call(data_iter_->second));
    return std::unique_ptr<value_type>(raw_ptr);
  }

  const reference operator*() const {
    return value_type(data_iter_->first, SharedUtil::Call(data_iter_->second));
  }

  _SharedPairIterator_ operator++(int) {
    _SharedPairIterator_ ret = *this;
    data_iter_++;
    return ret;
  }

  _SharedPairIterator_ operator++() {
    data_iter_++;
    return *this;
  }

  _SharedPairIterator_ operator--(int) {
    _SharedPairIterator_ ret = *this;
    data_iter_--;
    return ret;
  }

  _SharedPairIterator_ operator--() {
    data_iter_--;
    return *this;
  }

  pointer operator->() {
    auto* raw_ptr = new value_type(data_iter_->first, SharedUtil::Call(data_iter_->second));
    return std::unique_ptr<value_type>(raw_ptr);
  }

  reference operator*() {
    return value_type(data_iter_->first, SharedUtil::Call(data_iter_->second));
  }

 private:
  DataIter data_iter_;
};

template<typename T>
struct _SharedMutableUtil_ {
  template<typename DataT>
  static std::shared_ptr<T> Call(DataT& data) {
    return data.__SharedMutable__();
  }
};

template<typename MapT, typename T>
using _SharedMutPairIterator_ = _SharedPairIterator_<MapT, T, _SharedMutableUtil_<T>>;

template<typename T>
struct _SharedConstUtil_ {
  template<typename DataT>
  static std::shared_ptr<T> Call(DataT& data) {
    return data.__SharedConst__();
  }
};

template<typename MapT, typename T>
using _SharedConstPairIterator_ = _SharedPairIterator_<MapT, T, _SharedConstUtil_<T>>;

}  // namespace cfg
}  // namespace oneflow

#endif  // CFG_ONEFLOW_CFG_SHARED_PAIR_ITERATOR_H_
