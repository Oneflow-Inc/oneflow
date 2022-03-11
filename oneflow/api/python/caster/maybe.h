#include <pybind11/pybind11.h>
#include "oneflow/core/common/maybe.h"

namespace pybind11 {
namespace detail {

namespace {
template<typename T>
T& DeferenceIfSharedPtr(std::shared_ptr<T> ptr) {
  return *ptr;
}

template<typename T>
T&& DeferenceIfSharedPtr(T&& obj) {
  return std::forward<T>(obj);
}
}  // namespace

template<typename Type, typename Value = typename Type::ValueT>
struct maybe_caster {
  using value_conv = make_caster<Value>;

  template<typename T>
  static handle cast(T&& src, return_value_policy policy, handle parent) {
    if (!src.IsOk()) { oneflow::ThrowError(src.error()); }
    if (!std::is_lvalue_reference<T>::value) {
      policy = return_value_policy_override<Value>::policy(policy);
    }
    return value_conv::cast(DeferenceIfSharedPtr(CHECK_JUST(std::forward<T>(src))), policy, parent);
  }

  bool load(handle src, bool convert) {
    if (!src) { return false; }
    if (src.is_none()) {
      // does not accept `None` from Python. Users can use Optional in those cases.
      return false;
    }
    value_conv inner_caster;
    if (!inner_caster.load(src, convert)) { return false; }

    value = cast_op<Value&&>(std::move(inner_caster));
    return true;
  }

  PYBIND11_TYPE_CASTER(Type, _("Maybe[") + value_conv::name + _("]"));
};

template<>
struct maybe_caster<oneflow::Maybe<void>> {
  template<typename T>
  static handle cast(T&& src, return_value_policy policy, handle parent) {
    if (!src.IsOk()) { oneflow::ThrowError(src.error()); }
    return none().inc_ref();
  }

  bool load(handle src, bool convert) { return false; }

  PYBIND11_TYPE_CASTER(oneflow::Maybe<void>, _("Maybe[void]"));
};

template<typename T>
struct type_caster<oneflow::Maybe<T>> : public maybe_caster<oneflow::Maybe<T>> {};

}  // namespace detail
}  // namespace pybind11
