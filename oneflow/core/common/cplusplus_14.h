#ifndef ONEFLOW_CORE_COMMON_CPLUSPLUS_14_H_
#define ONEFLOW_CORE_COMMON_CPLUSPLUS_14_H_

#if __cplusplus < 201402L

namespace std {
template<class T>
struct unique_if {
  typedef unique_ptr<T> single_object;
};

template<class T>
struct unique_if<T[]> {
  typedef unique_ptr<T[]> unknown_bound;
};

template<class T, size_t N>
struct unique_if<T[N]> {
  typedef void known_bound;
};

template<class T, class... Args>
typename unique_if<T>::single_object make_unique(Args&&... args) {
  return unique_ptr<T>(new T(forward<Args>(args)...));
}

template<class T>
typename unique_if<T>::unknown_bound make_unique(size_t n) {
  typedef typename remove_extent<T>::type U;
  return unique_ptr<T>(new U[n]());
}

template<class T, class... Args>
typename unique_if<T>::known_bound make_unique(Args&&...) = delete;

template<typename T, T... Ints>
struct integer_sequence {
  static_assert(is_integral<T>::value, "");
  using value_type = T;
  static constexpr size_t size() { return sizeof...(Ints); }
};

template<typename T, T index, typename U, T... Ints>
struct make_integer_sequence_impl;

template<typename T, T index, T... Ints>
struct make_integer_sequence_impl<T, index, typename enable_if<index != 0>::type, Ints...>
    : make_integer_sequence_impl<T, index - 1, void, index, Ints...> {};

template<typename T, T index, T... Ints>
struct make_integer_sequence_impl<T, index, typename enable_if<index == 0>::type, Ints...> {
  using type = integer_sequence<T, index, Ints...>;
};

template<typename T, T n>
using make_integer_sequence = typename make_integer_sequence_impl<T, n - 1, void>::type;

template<size_t... Ints>
using index_sequence = integer_sequence<size_t, Ints...>;

template<size_t n>
using make_index_sequence = make_integer_sequence<size_t, n>;

template<typename... T>
using index_sequence_for = make_index_sequence<sizeof...(T)>;

}  // namespace std

#endif

#endif  // ONEFLOW_CORE_COMMON_CPLUSPLUS_14_H_
