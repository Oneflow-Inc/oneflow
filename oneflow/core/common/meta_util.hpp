#ifndef ONEFLOW_META_UTIL_HPP
#define ONEFLOW_META_UTIL_HPP

#include "oneflow/core/common/cplusplus_14.h"

namespace oneflow{

    template <typename... Args, typename Func, std::size_t... Idx>
    void for_each(const std::tuple<Args...>& t, Func&& f, std::index_sequence<Idx...>) {
        (void)std::initializer_list<int> { (f(std::get<Idx>(t)), void(), 0)...};
    }

    template <typename... Args, typename Func, std::size_t... Idx>
    void for_each_i(const std::tuple<Args...>& t, Func&& f, std::index_sequence<Idx...>) {
        (void)std::initializer_list<int> { (f(std::get<Idx>(t), std::integral_constant<size_t, Idx>{}), void(), 0)...};
    }

    template<typename T>
    using remove_const_reference_t = std::remove_const_t<std::remove_reference_t<T>>;

    template<size_t... Is>
    auto make_tuple_from_sequence(std::index_sequence<Is...>)->decltype(std::make_tuple(Is...)) {
        std::make_tuple(Is...);
    }

    template<size_t N>
    constexpr auto make_tuple_from_sequence()->decltype(make_tuple_from_sequence(std::make_index_sequence<N>{})) {
        return make_tuple_from_sequence(std::make_index_sequence<N>{});
    }

    namespace detail {
        template <class Tuple, class F, std::size_t...Is>
        void tuple_switch(const std::size_t i, Tuple&& t, F&& f, std::index_sequence<Is...>) {
            (void)std::initializer_list<int> {
                    (i == Is && (
                            (void)std::forward<F>(f)(std::integral_constant<size_t, Is>{}), 0))...
            };
        }
    } // namespace detail

    template <class Tuple, class F>
    inline void tuple_switch(const std::size_t i, Tuple&& t, F&& f) {
        constexpr auto N =
                std::tuple_size<std::remove_reference_t<Tuple>>::value;

        detail::tuple_switch(i, std::forward<Tuple>(t), std::forward<F>(f),
                             std::make_index_sequence<N>{});
    }
}

#endif //ONEFLOW_META_UTIL_HPP
