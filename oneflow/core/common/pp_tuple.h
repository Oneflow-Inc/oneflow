#ifndef ONEFLOW_CORE_COMMON_PP_TUPLE_H_
#define ONEFLOW_CORE_COMMON_PP_TUPLE_H_

#include "oneflow/core/common/pp_base.h"

#define OF_PP_TUPLE_PUSH_FRONT(tuple, x) \
  OF_PP_CAT(OF_PP_TUPLE_PUSH_FRONT_, OF_PP_IS_TUPLE_EMPTY(tuple))(tuple, x)

#define OF_PP_TUPLE_PUSH_FRONT_1(tuple, x) (x)
#define OF_PP_TUPLE_PUSH_FRONT_0(tuple, x) (x, OF_PP_TUPLE_TUPLE_TO_ARGS(tuple))

#define OF_PP_TUPLE_TUPLE_TO_ARGS(t) OF_PP_TUPLE_TUPLE_TO_ARGS_ t
#define OF_PP_TUPLE_TUPLE_TO_ARGS_(...) __VA_ARGS__

#define OF_PP_TUPLE_SIZE(tuple) \
  OF_PP_CAT(OF_PP_TUPLE_SIZE_, OF_PP_IS_TUPLE_EMPTY(tuple))(tuple)

#define OF_PP_TUPLE_SIZE_1(t) 0
#define OF_PP_TUPLE_SIZE_0(t) OF_PP_VARIADIC_SIZE t

#define OF_PP_VARIADIC_SIZE(...)                                             \
  OF_PP_VARIADIC_SIZE_I(__VA_ARGS__, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, \
                        54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42,  \
                        41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29,  \
                        28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,  \
                        15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, )
#define OF_PP_VARIADIC_SIZE_I(                                                 \
    e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, \
    e17, e18, e19, e20, e21, e22, e23, e24, e25, e26, e27, e28, e29, e30, e31, \
    e32, e33, e34, e35, e36, e37, e38, e39, e40, e41, e42, e43, e44, e45, e46, \
    e47, e48, e49, e50, e51, e52, e53, e54, e55, e56, e57, e58, e59, e60, e61, \
    e62, e63, size, ...)                                                       \
  size

#define OF_PP_IS_TUPLE_EMPTY(t) \
  OF_PP_IS_VARIADIC_EMPTY(OF_PP_TUPLE_TUPLE_TO_ARGS(t))

#define OF_PP_IS_VARIADIC_EMPTY(...)                                           \
  OF_PP_IS_VARIADIC_EMPTY_(/* test if there is just one argument, eventually   \
                              an empty one */                                  \
                           OF_PP_VARIADIC_HAS_COMMA(                           \
                               __VA_ARGS__), /* test if                        \
                                                _OF_PP_TRIGGER_PARENTHESIS_    \
                                                together with the argument     \
                                                adds a comma */                \
                           OF_PP_VARIADIC_HAS_COMMA(                           \
                               _OF_PP_TRIGGER_PARENTHESIS_                     \
                                   __VA_ARGS__), /* test if the argument       \
                                                    together with a            \
                                                    parenthesis adds a comma   \
                                                  */                           \
                           OF_PP_VARIADIC_HAS_COMMA(__VA_ARGS__(               \
                               /*empty*/)), /* test if placing it between      \
                                               _OF_PP_TRIGGER_PARENTHESIS_ and \
                                               the parenthesis adds a comma */ \
                           OF_PP_VARIADIC_HAS_COMMA(                           \
                               _OF_PP_TRIGGER_PARENTHESIS_ __VA_ARGS__(        \
                                   /*empty*/)))

#define OF_PP_IS_VARIADIC_EMPTY_(e0, e1, e2, e3) \
  OF_PP_VARIADIC_HAS_COMMA(OF_PP_CAT5(OF_PP_IS_EMPTY_CASE_, e0, e1, e2, e3))

#define OF_PP_VARIADIC_HAS_COMMA(...)                                          \
  OF_PP_VARIADIC_HAS_COMMA_I(                                                  \
      __VA_ARGS__, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  \
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0)
#define OF_PP_VARIADIC_HAS_COMMA_I(                                            \
    e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, \
    e17, e18, e19, e20, e21, e22, e23, e24, e25, e26, e27, e28, e29, e30, e31, \
    e32, e33, e34, e35, e36, e37, e38, e39, e40, e41, e42, e43, e44, e45, e46, \
    e47, e48, e49, e50, e51, e52, e53, e54, e55, e56, e57, e58, e59, e60, e61, \
    e62, e63, has_comma, ...)                                                  \
  has_comma

#define _OF_PP_TRIGGER_PARENTHESIS_(...) ,

#define OF_PP_CAT5(e0, e1, e2, e3, e4) e0##e1##e2##e3##e4
#define OF_PP_IS_EMPTY_CASE_0001 ,

#endif  // ONEFLOW_CORE_COMMON_PP_TUPLE_H_
