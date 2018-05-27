//
// Created by root on 5/27/18.
//

#ifndef ONEFLOW_META_UTIL_HPP
#define ONEFLOW_META_UTIL_HPP

namespace oneflow{
    template <typename T, T... Idx>
    struct integer_sequence{
        static_assert(std::is_integral<T>::value,
                      "integer_sequence<T, I...> requires T to be an integral type.");
        using value_type = T;
        static constexpr std::size_t size() noexcept { return sizeof...(Idx); }
    };

    template<size_t... Idx>
    using index_sequence = integer_sequence<size_t, Idx...>;

    template<size_t N, size_t... Idx>
    struct make_index_sequence_impl : make_index_sequence_impl<N-1, N-1, Idx...>{};

    template <size_t... Idx>
    struct make_index_sequence_impl<0, Idx...>{
        using type = index_sequence<Idx...>;
    };

    template<size_t N>
    using make_index_sequence = typename make_index_sequence_impl<N>::type;

    template <typename... Args, typename Func, std::size_t... Idx>
    void for_each(const std::tuple<Args...>& t, Func&& f, index_sequence<Idx...>) {
        (void)std::initializer_list<int> { (f(std::get<Idx>(t)), void(), 0)...};
    }

    template <typename... Args, typename Func, std::size_t... Idx>
    void for_each_i(const std::tuple<Args...>& t, Func&& f, index_sequence<Idx...>) {
        (void)std::initializer_list<int> { (f(std::get<Idx>(t), std::integral_constant<size_t, Idx>{}), void(), 0)...};
    }

    template<typename>
    struct array_size;

    template<typename T, size_t N>
    struct array_size<std::array<T,N> > {
        static size_t const size = N;
    };
}

#endif //ONEFLOW_META_UTIL_HPP
