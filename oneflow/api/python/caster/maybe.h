#include <pybind11/pybind11.h>

#include "oneflow/api/python/caster/common.h"
#include "oneflow/core/common/maybe.h"

namespace pybind11 {
namespace detail {

using oneflow::Maybe;

namespace impl {

template<typename T>
using IsHoldedInsideSharedPtrByMaybe =
    std::is_same<decltype(std::declval<Maybe<T>>()
                              .Data_YouAreNotAllowedToCallThisFuncOutsideThisFile()),
                 std::shared_ptr<T>>;

template<typename T, typename std::enable_if_t<IsSupportedByPybind11WhenInsideSharedPtr<T>::value
                                                   && IsHoldedInsideSharedPtrByMaybe<T>::value,
                                               int> = 0>
std::shared_ptr<T> GetOrThrowHelper(Maybe<T> x) {
  return x.GetPtrOrThrow();
}

template<typename T,
         typename std::enable_if_t<!IsSupportedByPybind11WhenInsideSharedPtr<T>::value || !IsHoldedInsideSharedPtrByMaybe<T>::value, int> = 0>
T GetOrThrowHelper(Maybe<T> x) {
  return x.GetOrThrow();
}

}  // namespace impl

template<typename Type>
struct maybe_caster {
  using Value = decltype(impl::GetOrThrowHelper(std::declval<Type>()));
  using value_conv = make_caster<Value>;

  template<typename T>
  static handle cast(T&& src, return_value_policy policy, handle parent) {
    if (!std::is_lvalue_reference<T>::value) {
      policy = return_value_policy_override<Value>::policy(policy);
    }
    return value_conv::cast(impl::GetOrThrowHelper(std::forward<T>(src)), policy, parent);
  }

  bool load(handle src, bool convert) {
    if (!src) { return false; }
    if (src.is_none()) {
      // Maybe<T> does not accept `None` from Python. Users can use Optional in those cases.
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
struct maybe_caster<Maybe<void>> {
  template<typename T>
  static handle cast(T&& src, return_value_policy policy, handle parent) {
    if (!src.IsOk()) { oneflow::ThrowError(src.error()); }
    return none().inc_ref();
  }

  bool load(handle src, bool convert) { return false; }

  PYBIND11_TYPE_CASTER(Maybe<void>, _("Maybe[void]"));
};

template<typename T>
struct type_caster<Maybe<T>> : public maybe_caster<Maybe<T>> {};

}  // namespace detail
}  // namespace pybind11
