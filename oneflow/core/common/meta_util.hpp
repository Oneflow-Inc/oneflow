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
    struct function_traits;

    template<typename Ret, typename... Args>
    struct function_traits<Ret(Args...)>
    {
    public:
        enum { arity = sizeof...(Args) };
        typedef Ret function_type(Args...);
        typedef Ret return_type;
        using stl_function_type = std::function<function_type>;
        typedef Ret(*pointer)(Args...);

        typedef std::tuple<Args...> tuple_type;
    };

    template<typename Ret, typename... Args>
    struct function_traits<Ret(*)(Args...)> : function_traits<Ret(Args...)>{};

    template <typename Ret, typename... Args>
    struct function_traits<std::function<Ret(Args...)>> : function_traits<Ret(Args...)>{};

    template <typename ReturnType, typename ClassType, typename... Args>
    struct function_traits<ReturnType(ClassType::*)(Args...)> : function_traits<ReturnType(Args...)>{};

    template <typename ReturnType, typename ClassType, typename... Args>
    struct function_traits<ReturnType(ClassType::*)(Args...) const> : function_traits<ReturnType(Args...)>{};

    template<typename Callable>
    struct function_traits : function_traits<decltype(&Callable::operator())>{};
}

#endif //ONEFLOW_META_UTIL_HPP
