#include <pybind11/pybind11.h>
#include "oneflow/core/common/optional.h"

namespace pybind11 {
namespace detail {

// Copy from pybind11 include/pybind11/stl.h
template<typename Type, typename Value = typename Type::value_type>
struct oneflow_optional_caster {
  using value_conv = make_caster<Value>;

  template<typename T>
  static handle cast(T&& src, return_value_policy policy, handle parent) {
    if (!src) { return none().inc_ref(); }
    if (!std::is_lvalue_reference<T>::value) {
      policy = return_value_policy_override<Value>::policy(policy);
    }
    return value_conv::cast(CHECK_JUST(std::forward<T>(src)), policy, parent);
  }

  bool load(handle src, bool convert) {
    if (!src) { return false; }
    if (src.is_none()) {
      return true;  // default-constructed value is already empty
    }
    value_conv inner_caster;
    if (!inner_caster.load(src, convert)) { return false; }

    value = cast_op<Value&&>(std::move(inner_caster));
    return true;
  }

  PYBIND11_TYPE_CASTER(Type, _("Optional[") + value_conv::name + _("]"));
};

template<typename T>
struct type_caster<oneflow::Optional<T>> : public oneflow_optional_caster<oneflow::Optional<T>> {};

}  // namespace detail
}  // namespace pybind11
