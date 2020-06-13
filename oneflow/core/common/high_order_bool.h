#ifndef ONEFLOW_CORE_COMMON_HIGH_ORDER_BOOL_H_
#define ONEFLOW_CORE_COMMON_HIGH_ORDER_BOOL_H_

#include <string>
#include <memory>
#include <sstream>

namespace oneflow {

namespace hob {

template<typename T>
class BoolFunctor {
 public:
  virtual ~BoolFunctor() {}
  virtual bool operator()(const T& ctx) const = 0;
  virtual std::string DebugStr(const T& ctx, bool display_result = true) const = 0;

 protected:
  BoolFunctor() = default;
};

template<typename T>
class BoolFunctorPtr final {
 public:
  BoolFunctorPtr() = default;
  BoolFunctorPtr(const BoolFunctorPtr&) = default;
  BoolFunctorPtr(BoolFunctorPtr&&) = default;
  ~BoolFunctorPtr(){};
  BoolFunctorPtr(const std::shared_ptr<const BoolFunctor<T>>& ptr) : ptr_(ptr) {}
  std::string DebugStr(const T& ctx, bool display_result = true) const;
  bool operator()(const T& ctx) const;
  BoolFunctorPtr operator&(const BoolFunctorPtr& ptr) const;
  BoolFunctorPtr operator|(const BoolFunctorPtr& ptr) const;
  BoolFunctorPtr operator~() const;
  BoolFunctorPtr& operator=(BoolFunctorPtr& ptr) {
    this->ptr_ = ptr.ptr_;
    return *this;
  }

 private:
  std::shared_ptr<const BoolFunctor<T>> ptr_;
};

template<typename T>
class AndBoolFunctor final : public BoolFunctor<T> {
 public:
  AndBoolFunctor() = delete;
  AndBoolFunctor(const BoolFunctorPtr<T> lhs, const BoolFunctorPtr<T> rhs) : lhs_(lhs), rhs_(rhs) {}
  ~AndBoolFunctor() override = default;

  std::string DebugStr(const T& ctx, bool display_result) const override {
    std::string l_str = lhs_.DebugStr(ctx, display_result);
    display_result = display_result && lhs_(ctx);
    std::string r_str = rhs_.DebugStr(ctx, display_result);
    std::ostringstream string_stream;
    string_stream << "(" << l_str << " and " << r_str << ")";
    return string_stream.str();
  }

  bool operator()(const T& ctx) const override { return lhs_(ctx) && rhs_(ctx); }

 private:
  const BoolFunctorPtr<T> lhs_;
  const BoolFunctorPtr<T> rhs_;
};

template<typename T>
class OrBoolFunctor final : public BoolFunctor<T> {
 public:
  OrBoolFunctor() = delete;
  OrBoolFunctor(const BoolFunctorPtr<T> lhs, const BoolFunctorPtr<T> rhs) : lhs_(lhs), rhs_(rhs) {}
  ~OrBoolFunctor() override = default;

  std::string DebugStr(const T& ctx, bool display_result) const override {
    std::string l_str = lhs_.DebugStr(ctx, display_result);
    display_result = display_result && (!lhs_(ctx));
    std::string r_str = rhs_.DebugStr(ctx, display_result);
    std::ostringstream string_stream;
    string_stream << "(" << l_str << " or " << r_str << ")";
    return string_stream.str();
  }

  bool operator()(const T& ctx) const override { return lhs_(ctx) || rhs_(ctx); }

 private:
  const BoolFunctorPtr<T> lhs_;
  const BoolFunctorPtr<T> rhs_;
};

template<typename T>
class NotBoolFunctor final : public BoolFunctor<T> {
 public:
  NotBoolFunctor() = delete;
  NotBoolFunctor(const BoolFunctorPtr<T> hs) : hs_(hs) {}
  ~NotBoolFunctor() override = default;

  std::string DebugStr(const T& ctx, bool display_result) const override {
    std::ostringstream string_stream;
    string_stream << "("
                  << "not " << hs_.DebugStr(ctx, display_result) << ")";
    return string_stream.str();
  }

  bool operator()(const T& ctx) const override { return !hs_(ctx); }

 private:
  const BoolFunctorPtr<T> hs_;
};

template<typename T>
BoolFunctorPtr<T> BoolFunctorPtr<T>::operator&(const BoolFunctorPtr& ptr) const {
  std::shared_ptr<const BoolFunctor<T>> and_ptr =
      std::make_shared<const AndBoolFunctor<T>>(this->ptr_, ptr.ptr_);
  return BoolFunctorPtr<T>(and_ptr);
}

template<typename T>
BoolFunctorPtr<T> BoolFunctorPtr<T>::operator|(const BoolFunctorPtr& ptr) const {
  std::shared_ptr<const BoolFunctor<T>> or_ptr =
      std::make_shared<const OrBoolFunctor<T>>(this->ptr_, ptr.ptr_);
  return BoolFunctorPtr<T>(or_ptr);
}

template<typename T>
BoolFunctorPtr<T> BoolFunctorPtr<T>::operator~() const {
  std::shared_ptr<const BoolFunctor<T>> not_ptr =
      std::make_shared<const NotBoolFunctor<T>>(this->ptr_);
  return BoolFunctorPtr<T>(not_ptr);
}

template<typename T>
std::string BoolFunctorPtr<T>::DebugStr(const T& ctx, bool display_result) const {
  return this->ptr_->DebugStr(ctx, display_result);
}

template<typename T>
bool BoolFunctorPtr<T>::operator()(const T& ctx) const {
  return (*this->ptr_)(ctx);
}

}  // namespace hob

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_HIGH_ORDER_BOOL_H_
