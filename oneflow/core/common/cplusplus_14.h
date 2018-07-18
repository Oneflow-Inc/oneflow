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

template<size_t... Ints>
struct index_sequence {
  using type = index_sequence;
  using value_type = size_t;
  static constexpr std::size_t size() noexcept { return sizeof...(Ints); }
};

// --------------------------------------------------------------

template<class Sequence1, class Sequence2>
struct _merge_and_renumber;

template<size_t... I1, size_t... I2>
struct _merge_and_renumber<index_sequence<I1...>, index_sequence<I2...>>
    : index_sequence<I1..., (sizeof...(I1) + I2)...> {};

// --------------------------------------------------------------

template<size_t N>
struct make_index_sequence : _merge_and_renumber<typename make_index_sequence<N / 2>::type,
                                                 typename make_index_sequence<N - N / 2>::type> {};

template<>
struct make_index_sequence<0> : index_sequence<> {};
template<>
struct make_index_sequence<1> : index_sequence<0> {};

template<typename... T>
using index_sequence_for = make_index_sequence<sizeof...(T)>;

template<bool B, class T = void>
using enable_if_t = typename enable_if<B, T>::type;

template<typename T>
using remove_const_t = typename remove_const<T>::type;

template<typename T>
using remove_reference_t = typename remove_reference<T>::type;

}  // namespace std

#endif

#endif  // ONEFLOW_CORE_COMMON_CPLUSPLUS_14_H_
