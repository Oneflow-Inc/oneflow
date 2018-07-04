#ifndef ONEFLOW_CORE_COMMON_PREPROCESSOR_INTERNAL_H_
#define ONEFLOW_CORE_COMMON_PREPROCESSOR_INTERNAL_H_

// Base

#define OF_PP_INTERNAL_STRINGIZE(text) OF_PP_INTERNAL_STRINGIZE_I(text)
#define OF_PP_INTERNAL_STRINGIZE_I(text) #text

#define OF_PP_INTERNAL_CAT(a, b) OF_PP_INTERNAL_CAT_I(a, b)
#define OF_PP_INTERNAL_CAT_I(a, b) a##b

#define OF_PP_INTERNAL_JOIN(glue, ...)                                                     \
  OF_PP_INTERNAL_CAT(                                                                      \
      OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_JOIN_, OF_PP_INTERNAL_VARIADIC_SIZE(__VA_ARGS__))( \
          glue, __VA_ARGS__), )

#define OF_PP_INTERNAL_JOIN_0(glue)
#define OF_PP_INTERNAL_JOIN_1(glue, x) x
#define OF_PP_INTERNAL_JOIN_2(glue, x, ...) \
  OF_PP_INTERNAL_CAT(                       \
      OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_CAT(x, glue), OF_PP_INTERNAL_JOIN_1(glue, __VA_ARGS__)), )
#define OF_PP_INTERNAL_JOIN_3(glue, x, ...) \
  OF_PP_INTERNAL_CAT(                       \
      OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_CAT(x, glue), OF_PP_INTERNAL_JOIN_2(glue, __VA_ARGS__)), )
#define OF_PP_INTERNAL_JOIN_4(glue, x, ...) \
  OF_PP_INTERNAL_CAT(                       \
      OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_CAT(x, glue), OF_PP_INTERNAL_JOIN_3(glue, __VA_ARGS__)), )
#define OF_PP_INTERNAL_JOIN_5(glue, x, ...) \
  OF_PP_INTERNAL_CAT(                       \
      OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_CAT(x, glue), OF_PP_INTERNAL_JOIN_4(glue, __VA_ARGS__)), )
#define OF_PP_INTERNAL_JOIN_6(glue, x, ...) \
  OF_PP_INTERNAL_CAT(                       \
      OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_CAT(x, glue), OF_PP_INTERNAL_JOIN_5(glue, __VA_ARGS__)), )
#define OF_PP_INTERNAL_JOIN_7(glue, x, ...) \
  OF_PP_INTERNAL_CAT(                       \
      OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_CAT(x, glue), OF_PP_INTERNAL_JOIN_6(glue, __VA_ARGS__)), )
#define OF_PP_INTERNAL_JOIN_8(glue, x, ...) \
  OF_PP_INTERNAL_CAT(                       \
      OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_CAT(x, glue), OF_PP_INTERNAL_JOIN_7(glue, __VA_ARGS__)), )
#define OF_PP_INTERNAL_JOIN_9(glue, x, ...) \
  OF_PP_INTERNAL_CAT(                       \
      OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_CAT(x, glue), OF_PP_INTERNAL_JOIN_8(glue, __VA_ARGS__)), )
#define OF_PP_INTERNAL_JOIN_10(glue, x, ...) \
  OF_PP_INTERNAL_CAT(                        \
      OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_CAT(x, glue), OF_PP_INTERNAL_JOIN_9(glue, __VA_ARGS__)), )
#define OF_PP_INTERNAL_JOIN_11(glue, x, ...)                         \
  OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_CAT(x, glue), \
                                        OF_PP_INTERNAL_JOIN_10(glue, __VA_ARGS__)), )
#define OF_PP_INTERNAL_JOIN_12(glue, x, ...)                         \
  OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_CAT(x, glue), \
                                        OF_PP_INTERNAL_JOIN_11(glue, __VA_ARGS__)), )
#define OF_PP_INTERNAL_JOIN_13(glue, x, ...)                         \
  OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_CAT(x, glue), \
                                        OF_PP_INTERNAL_JOIN_12(glue, __VA_ARGS__)), )
#define OF_PP_INTERNAL_JOIN_14(glue, x, ...)                         \
  OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_CAT(x, glue), \
                                        OF_PP_INTERNAL_JOIN_13(glue, __VA_ARGS__)), )
#define OF_PP_INTERNAL_JOIN_15(glue, x, ...)                         \
  OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_CAT(x, glue), \
                                        OF_PP_INTERNAL_JOIN_14(glue, __VA_ARGS__)), )

#define OF_PP_INTERNAL_SEQ_HEAD(seq) OF_PP_INTERNAL_PAIR_FIRST(OF_PP_INTERNAL_SEQ_TO_PAIR(seq))
#define OF_PP_INTERNAL_SEQ_TAIL(seq) OF_PP_INTERNAL_PAIR_SECOND(OF_PP_INTERNAL_SEQ_TO_PAIR(seq))

#define OF_PP_INTERNAL_SEQ_TO_PAIR(seq) (OF_PP_INTERNAL_SEQ_TO_PAIR_ seq)
#define OF_PP_INTERNAL_SEQ_TO_PAIR_(x) x, OF_PP_INTERNAL_NIL
#define OF_PP_INTERNAL_NIL

#define OF_PP_INTERNAL_PAIR_FIRST(t) OF_PP_INTERNAL_PAIR_FIRST_I(t)
#define OF_PP_INTERNAL_PAIR_FIRST_I(t) OF_PP_INTERNAL_FIRST_ARG t
#define OF_PP_INTERNAL_PAIR_SECOND(t) OF_PP_INTERNAL_PAIR_SECOND_I(t)
#define OF_PP_INTERNAL_PAIR_SECOND_I(t) OF_PP_INTERNAL_SECOND_ARG t

#define OF_PP_INTERNAL_FIRST_ARG(x, ...) x
#define OF_PP_INTERNAL_SECOND_ARG(x, y, ...) y

#define OF_PP_INTERNAL_MAKE_TUPLE(...) (__VA_ARGS__)
#define OF_PP_INTERNAL_MAKE_TUPLE_SEQ(...) (OF_PP_INTERNAL_MAKE_TUPLE(__VA_ARGS__))

// Tuple

#define OF_PP_INTERNAL_TUPLE_PUSH_FRONT(tuple, x)                                        \
  OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_TUPLE_PUSH_FRONT_, OF_PP_INTERNAL_TUPLE_SIZE(tuple)) \
  (tuple, x)

#define OF_PP_INTERNAL_TUPLE_PUSH_FRONT_0(tuple, x) (x)
#define OF_PP_INTERNAL_TUPLE_PUSH_FRONT_1(tuple, x) (x, OF_PP_INTERNAL_TUPLE_ELEM(0, tuple))
#define OF_PP_INTERNAL_TUPLE_PUSH_FRONT_2(tuple, x) \
  (x, OF_PP_INTERNAL_TUPLE_ELEM(0, tuple), OF_PP_INTERNAL_TUPLE_ELEM(1, tuple))
#define OF_PP_INTERNAL_TUPLE_PUSH_FRONT_3(tuple, x)                             \
  (x, OF_PP_INTERNAL_TUPLE_ELEM(0, tuple), OF_PP_INTERNAL_TUPLE_ELEM(1, tuple), \
   OF_PP_INTERNAL_TUPLE_ELEM(2, tuple))
#define OF_PP_INTERNAL_TUPLE_PUSH_FRONT_4(tuple, x)                             \
  (x, OF_PP_INTERNAL_TUPLE_ELEM(0, tuple), OF_PP_INTERNAL_TUPLE_ELEM(1, tuple), \
   OF_PP_INTERNAL_TUPLE_ELEM(2, tuple), OF_PP_INTERNAL_TUPLE_ELEM(3, tuple))
#define OF_PP_INTERNAL_TUPLE_PUSH_FRONT_5(tuple, x)                             \
  (x, OF_PP_INTERNAL_TUPLE_ELEM(0, tuple), OF_PP_INTERNAL_TUPLE_ELEM(1, tuple), \
   OF_PP_INTERNAL_TUPLE_ELEM(2, tuple), OF_PP_INTERNAL_TUPLE_ELEM(3, tuple),    \
   OF_PP_INTERNAL_TUPLE_ELEM(4, tuple))
#define OF_PP_INTERNAL_TUPLE_PUSH_FRONT_6(tuple, x)                             \
  (x, OF_PP_INTERNAL_TUPLE_ELEM(0, tuple), OF_PP_INTERNAL_TUPLE_ELEM(1, tuple), \
   OF_PP_INTERNAL_TUPLE_ELEM(2, tuple), OF_PP_INTERNAL_TUPLE_ELEM(3, tuple),    \
   OF_PP_INTERNAL_TUPLE_ELEM(4, tuple), OF_PP_INTERNAL_TUPLE_ELEM(5, tuple))
#define OF_PP_INTERNAL_TUPLE_PUSH_FRONT_7(tuple, x)                             \
  (x, OF_PP_INTERNAL_TUPLE_ELEM(0, tuple), OF_PP_INTERNAL_TUPLE_ELEM(1, tuple), \
   OF_PP_INTERNAL_TUPLE_ELEM(2, tuple), OF_PP_INTERNAL_TUPLE_ELEM(3, tuple),    \
   OF_PP_INTERNAL_TUPLE_ELEM(4, tuple), OF_PP_INTERNAL_TUPLE_ELEM(5, tuple),    \
   OF_PP_INTERNAL_TUPLE_ELEM(6, tuple))
#define OF_PP_INTERNAL_TUPLE_PUSH_FRONT_8(tuple, x)                             \
  (x, OF_PP_INTERNAL_TUPLE_ELEM(0, tuple), OF_PP_INTERNAL_TUPLE_ELEM(1, tuple), \
   OF_PP_INTERNAL_TUPLE_ELEM(2, tuple), OF_PP_INTERNAL_TUPLE_ELEM(3, tuple),    \
   OF_PP_INTERNAL_TUPLE_ELEM(4, tuple), OF_PP_INTERNAL_TUPLE_ELEM(5, tuple),    \
   OF_PP_INTERNAL_TUPLE_ELEM(6, tuple), OF_PP_INTERNAL_TUPLE_ELEM(7, tuple))

#define OF_PP_INTERNAL_TUPLE_ELEM(n, t) OF_PP_INTERNAL_TUPLE_ELEM_I(n, t)
#define OF_PP_INTERNAL_TUPLE_ELEM_I(n, t) \
  OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_ARG_, n) t, )

#define OF_PP_INTERNAL_ARG_0(a0, ...) a0
#define OF_PP_INTERNAL_ARG_1(a0, a1, ...) a1
#define OF_PP_INTERNAL_ARG_2(a0, a1, a2, ...) a2
#define OF_PP_INTERNAL_ARG_3(a0, a1, a2, a3, ...) a3
#define OF_PP_INTERNAL_ARG_4(a0, a1, a2, a3, a4, ...) a4
#define OF_PP_INTERNAL_ARG_5(a0, a1, a2, a3, a4, a5, ...) a5
#define OF_PP_INTERNAL_ARG_6(a0, a1, a2, a3, a4, a5, a6, ...) a6
#define OF_PP_INTERNAL_ARG_7(a0, a1, a2, a3, a4, a5, a6, a7, ...) a7

#define OF_PP_INTERNAL_TUPLE_SIZE(tuple)                                               \
  OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_TUPLE_SIZE_, OF_PP_INTERNAL_IS_TUPLE_EMPTY(tuple)) \
  (tuple)

#define OF_PP_INTERNAL_TUPLE_SIZE_1(t) 0
#define OF_PP_INTERNAL_TUPLE_SIZE_0(t) OF_PP_INTERNAL_TUPLE_SIZE_0_I(t)
#define OF_PP_INTERNAL_TUPLE_SIZE_0_I(t) OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_VARIADIC_SIZE t, )

#define OF_PP_INTERNAL_VARIADIC_SIZE(...)                                                          \
  OF_PP_INTERNAL_CAT(                                                                              \
      OF_PP_INTERNAL_VARIADIC_SIZE_I(                                                              \
          __VA_ARGS__, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, \
          45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24,  \
          23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, ), )
#define OF_PP_INTERNAL_VARIADIC_SIZE_I(                                                            \
    e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, e17, e18, e19, e20, \
    e21, e22, e23, e24, e25, e26, e27, e28, e29, e30, e31, e32, e33, e34, e35, e36, e37, e38, e39, \
    e40, e41, e42, e43, e44, e45, e46, e47, e48, e49, e50, e51, e52, e53, e54, e55, e56, e57, e58, \
    e59, e60, e61, e62, e63, size, ...)                                                            \
  size

#define OF_PP_INTERNAL_IS_TUPLE_EMPTY(t) OF_PP_INTERNAL_IS_TUPLE_EMPTY_I(t)
#define OF_PP_INTERNAL_IS_TUPLE_EMPTY_I(t) OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_IS_VARIADIC_EMPTY t, )

#define OF_PP_INTERNAL_IS_VARIADIC_EMPTY(...)                                                 \
  OF_PP_INTERNAL_IS_VARIADIC_EMPTY_(/* test if there is just one argument,                    \
                              eventually an empty one */                                      \
                                    OF_PP_INTERNAL_VARIADIC_HAS_COMMA(                        \
                                        __VA_ARGS__), /* test if                              \
                                                         _OF_PP_INTERNAL_TRIGGER_PARENTHESIS_ \
                                                         together with the                    \
                                                         argument adds a comma                \
                                                       */                                     \
                                    OF_PP_INTERNAL_VARIADIC_HAS_COMMA(                        \
                                        _OF_PP_INTERNAL_TRIGGER_PARENTHESIS_                  \
                                            __VA_ARGS__), /* test if the                      \
                                                             argument together                \
                                                             with a                           \
                                                             parenthesis adds                 \
                                                             a comma                          \
                                                           */                                 \
                                    OF_PP_INTERNAL_VARIADIC_HAS_COMMA(__VA_ARGS__(            \
                                        /*empty*/)), /* test if placing it                    \
                                                        between                               \
                                                        _OF_PP_INTERNAL_TRIGGER_PARENTHESIS_  \
                                                        and the                               \
                                                        parenthesis adds a                    \
                                                        comma */                              \
                                    OF_PP_INTERNAL_VARIADIC_HAS_COMMA(                        \
                                        _OF_PP_INTERNAL_TRIGGER_PARENTHESIS_ __VA_ARGS__(     \
                                            /*empty*/)))

#define OF_PP_INTERNAL_IS_VARIADIC_EMPTY_(e0, e1, e2, e3) \
  OF_PP_INTERNAL_VARIADIC_HAS_COMMA(                      \
      OF_PP_INTERNAL_CAT5(OF_PP_INTERNAL_IS_EMPTY_CASE_, e0, e1, e2, e3))

#define OF_PP_INTERNAL_VARIADIC_HAS_COMMA(...)                                                    \
  OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_VARIADIC_HAS_COMMA_I(                                         \
                         __VA_ARGS__, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  \
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0), )
#define OF_PP_INTERNAL_VARIADIC_HAS_COMMA_I(                                                       \
    e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, e17, e18, e19, e20, \
    e21, e22, e23, e24, e25, e26, e27, e28, e29, e30, e31, e32, e33, e34, e35, e36, e37, e38, e39, \
    e40, e41, e42, e43, e44, e45, e46, e47, e48, e49, e50, e51, e52, e53, e54, e55, e56, e57, e58, \
    e59, e60, e61, e62, e63, has_comma, ...)                                                       \
  has_comma

#define _OF_PP_INTERNAL_TRIGGER_PARENTHESIS_(...) ,

#define OF_PP_INTERNAL_CAT5(e0, e1, e2, e3, e4) e0##e1##e2##e3##e4
#define OF_PP_INTERNAL_IS_EMPTY_CASE_0001 ,

// Seq Product

#define OF_PP_INTERNAL_SEQ_PRODUCT_FOR_EACH_TUPLE(macro, seq0, ...) \
  OF_PP_INTERNAL_SEQ_FOR_EACH_TUPLE(macro, _, OF_PP_INTERNAL_SEQ_PRODUCT(seq0, __VA_ARGS__))

#define OF_PP_INTERNAL_SEQ_PRODUCT(seq0, ...)         \
  OF_PP_INTERNAL_CAT(                                 \
      OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_SEQ_PRODUCT_, \
                         OF_PP_INTERNAL_VARIADIC_SIZE(seq0, __VA_ARGS__))(seq0, __VA_ARGS__), )

#define OF_PP_INTERNAL_SEQ_PRODUCT_0()
#define OF_PP_INTERNAL_SEQ_PRODUCT_1(seq0) OF_PP_INTERNAL_TUPLE_SEQ_X_ATOMIC_SEQ((()), seq0)
#define OF_PP_INTERNAL_SEQ_PRODUCT_2(seq0, ...) \
  OF_PP_INTERNAL_TUPLE_SEQ_X_ATOMIC_SEQ(OF_PP_INTERNAL_SEQ_PRODUCT_1(__VA_ARGS__), seq0)
#define OF_PP_INTERNAL_SEQ_PRODUCT_3(seq0, ...) \
  OF_PP_INTERNAL_TUPLE_SEQ_X_ATOMIC_SEQ(OF_PP_INTERNAL_SEQ_PRODUCT_2(__VA_ARGS__), seq0)
#define OF_PP_INTERNAL_SEQ_PRODUCT_4(seq0, ...) \
  OF_PP_INTERNAL_TUPLE_SEQ_X_ATOMIC_SEQ(OF_PP_INTERNAL_SEQ_PRODUCT_3(__VA_ARGS__), seq0)
#define OF_PP_INTERNAL_SEQ_PRODUCT_5(seq0, ...) \
  OF_PP_INTERNAL_TUPLE_SEQ_X_ATOMIC_SEQ(OF_PP_INTERNAL_SEQ_PRODUCT_4(__VA_ARGS__), seq0)
#define OF_PP_INTERNAL_SEQ_PRODUCT_6(seq0, ...) \
  OF_PP_INTERNAL_TUPLE_SEQ_X_ATOMIC_SEQ(OF_PP_INTERNAL_SEQ_PRODUCT_5(__VA_ARGS__), seq0)
#define OF_PP_INTERNAL_SEQ_PRODUCT_7(seq0, ...) \
  OF_PP_INTERNAL_TUPLE_SEQ_X_ATOMIC_SEQ(OF_PP_INTERNAL_SEQ_PRODUCT_6(__VA_ARGS__), seq0)
#define OF_PP_INTERNAL_SEQ_PRODUCT_8(seq0, ...) \
  OF_PP_INTERNAL_TUPLE_SEQ_X_ATOMIC_SEQ(OF_PP_INTERNAL_SEQ_PRODUCT_7(__VA_ARGS__), seq0)
#define OF_PP_INTERNAL_SEQ_PRODUCT_9(seq0, ...) \
  OF_PP_INTERNAL_TUPLE_SEQ_X_ATOMIC_SEQ(OF_PP_INTERNAL_SEQ_PRODUCT_8(__VA_ARGS__), seq0)
#define OF_PP_INTERNAL_SEQ_PRODUCT_10(seq0, ...) \
  OF_PP_INTERNAL_TUPLE_SEQ_X_ATOMIC_SEQ(OF_PP_INTERNAL_SEQ_PRODUCT_9(__VA_ARGS__), seq0)

// Seq ForEach

#define OF_PP_INTERNAL_FOR_EACH_TUPLE(macro, seq) OF_PP_INTERNAL_SEQ_FOR_EACH_TUPLE(macro, _, seq)
#define OF_PP_INTERNAL_TUPLE_SEQ_X_ATOMIC_SEQ(tuple_seq, atomic_seq)       \
  OF_PP_INTERNAL_D1_SEQ_FOR_EACH(OF_PP_INTERNAL_D1_APPLY_ATOMIC_WITH_DATA, \
                                 OF_PP_INTERNAL_TUPLE_X_ATOMIC_SEQ, atomic_seq, tuple_seq)

#define OF_PP_INTERNAL_TUPLE_X_ATOMIC_SEQ(atomic_seq, tuple)               \
  OF_PP_INTERNAL_D2_SEQ_FOR_EACH(OF_PP_INTERNAL_D2_APPLY_ATOMIC_WITH_DATA, \
                                 OF_PP_INTERNAL_MAKE_SEQ_TUPLE_PUSH_FRONT, tuple, atomic_seq)

#define OF_PP_INTERNAL_D1_APPLY_ATOMIC_WITH_DATA(m, d, x) m(d, x)
#define OF_PP_INTERNAL_D2_APPLY_ATOMIC_WITH_DATA(m, d, x) m(d, x)

#define OF_PP_INTERNAL_MAKE_SEQ_TUPLE_PUSH_FRONT(tuple, x) \
  (OF_PP_INTERNAL_TUPLE_PUSH_FRONT(tuple, x))

// Seq Size

#define OF_PP_INTERNAL_SEQ_SIZE(seq) OF_PP_INTERNAL_SEQ_SIZE_I(seq)
#define OF_PP_INTERNAL_SEQ_SIZE_I(seq) \
  OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_SEQ_SIZE_, OF_PP_INTERNAL_SEQ_SIZE_0 seq)

#define OF_PP_INTERNAL_SEQ_FOR_EACH_TUPLE OF_PP_INTERNAL_D1_SEQ_FOR_EACH_TUPLE

#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_TUPLE(m, d, seq) \
  OF_PP_INTERNAL_D1_SEQ_FOR_EACH(OF_PP_INTERNAL_APPLY_TUPLE, m, d, seq)
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_TUPLE(m, d, seq) \
  OF_PP_INTERNAL_D2_SEQ_FOR_EACH(OF_PP_INTERNAL_APPLY_TUPLE, m, d, seq)

#define OF_PP_INTERNAL_SEQ_FOR_EACH_ATOMIC OF_PP_INTERNAL_D1_SEQ_FOR_EACH_ATOMIC

#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_ATOMIC(m, d, seq) \
  OF_PP_INTERNAL_D1_SEQ_FOR_EACH(OF_PP_INTERNAL_APPLY_ATOMIC, m, d, seq)
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_ATOMIC(m, d, seq) \
  OF_PP_INTERNAL_D2_SEQ_FOR_EACH(OF_PP_INTERNAL_APPLY_ATOMIC, m, d, seq)

#define OF_PP_INTERNAL_APPLY_TUPLE(m, d, t) OF_PP_INTERNAL_APPLY_TUPLE_I(m, d, t)
#define OF_PP_INTERNAL_APPLY_TUPLE_I(m, d, t) m t
#define OF_PP_INTERNAL_APPLY_ATOMIC(m, d, x) m(x)
#define OF_PP_INTERNAL_APPLY_ATOMIC_WITH_DATA(m, d, x) m(d, x)

#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH(apply, m, d, seq)                            \
  OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_D1_SEQ_FOR_EACH_, OF_PP_INTERNAL_SEQ_SIZE(seq)) \
  (apply, m, d, seq)

#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH(apply, m, d, seq)                            \
  OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_D2_SEQ_FOR_EACH_, OF_PP_INTERNAL_SEQ_SIZE(seq)) \
  (apply, m, d, seq)

#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_0(apply, m, d, seq)
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_0(apply, m, d, seq)

//  php code to generate iterator macro
/*
<?php $limit = 256; for ($i = 0; $i < $limit; ++$i) {?>
#define OF_PP_INTERNAL_SEQ_SIZE_<?= $i?>(_) \
  OF_PP_INTERNAL_SEQ_SIZE_<?= $i + 1?>

#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_<?= $i?> <?= $i?>

<?php $dim = 2; for ($d = 1; $d <= $dim; ++$d) {?>
#define OF_PP_INTERNAL_D<?= $d?>_SEQ_FOR_EACH_<?= $i + 1?>(apply, m, d, seq) \
    apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) \
      OF_PP_INTERNAL_D<?= $d?>_SEQ_FOR_EACH_<?= $i?>( \
            apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
<?php }?>

<?php }?>
*/

//  do not edit iterator macro directly, it's generated by the above php code.
#define OF_PP_INTERNAL_SEQ_SIZE_0(_) OF_PP_INTERNAL_SEQ_SIZE_1
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_0 0
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_1(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_0(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_1(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_0(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_1(_) OF_PP_INTERNAL_SEQ_SIZE_2
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_1 1
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_2(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_1(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_2(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_1(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_2(_) OF_PP_INTERNAL_SEQ_SIZE_3
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_2 2
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_3(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_2(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_3(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_2(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_3(_) OF_PP_INTERNAL_SEQ_SIZE_4
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_3 3
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_4(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_3(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_4(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_3(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_4(_) OF_PP_INTERNAL_SEQ_SIZE_5
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_4 4
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_5(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_4(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_5(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_4(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_5(_) OF_PP_INTERNAL_SEQ_SIZE_6
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_5 5
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_6(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_5(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_6(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_5(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_6(_) OF_PP_INTERNAL_SEQ_SIZE_7
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_6 6
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_7(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_6(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_7(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_6(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_7(_) OF_PP_INTERNAL_SEQ_SIZE_8
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_7 7
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_8(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_7(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_8(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_7(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_8(_) OF_PP_INTERNAL_SEQ_SIZE_9
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_8 8
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_9(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_8(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_9(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_8(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_9(_) OF_PP_INTERNAL_SEQ_SIZE_10
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_9 9
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_10(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_9(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_10(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_9(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_10(_) OF_PP_INTERNAL_SEQ_SIZE_11
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_10 10
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_11(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_10(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_11(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_10(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_11(_) OF_PP_INTERNAL_SEQ_SIZE_12
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_11 11
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_12(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_11(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_12(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_11(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_12(_) OF_PP_INTERNAL_SEQ_SIZE_13
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_12 12
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_13(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_12(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_13(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_12(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_13(_) OF_PP_INTERNAL_SEQ_SIZE_14
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_13 13
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_14(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_13(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_14(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_13(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_14(_) OF_PP_INTERNAL_SEQ_SIZE_15
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_14 14
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_15(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_14(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_15(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_14(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_15(_) OF_PP_INTERNAL_SEQ_SIZE_16
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_15 15
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_16(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_15(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_16(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_15(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_16(_) OF_PP_INTERNAL_SEQ_SIZE_17
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_16 16
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_17(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_16(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_17(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_16(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_17(_) OF_PP_INTERNAL_SEQ_SIZE_18
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_17 17
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_18(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_17(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_18(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_17(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_18(_) OF_PP_INTERNAL_SEQ_SIZE_19
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_18 18
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_19(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_18(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_19(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_18(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_19(_) OF_PP_INTERNAL_SEQ_SIZE_20
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_19 19
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_20(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_19(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_20(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_19(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_20(_) OF_PP_INTERNAL_SEQ_SIZE_21
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_20 20
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_21(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_20(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_21(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_20(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_21(_) OF_PP_INTERNAL_SEQ_SIZE_22
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_21 21
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_22(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_21(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_22(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_21(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_22(_) OF_PP_INTERNAL_SEQ_SIZE_23
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_22 22
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_23(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_22(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_23(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_22(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_23(_) OF_PP_INTERNAL_SEQ_SIZE_24
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_23 23
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_24(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_23(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_24(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_23(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_24(_) OF_PP_INTERNAL_SEQ_SIZE_25
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_24 24
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_25(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_24(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_25(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_24(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_25(_) OF_PP_INTERNAL_SEQ_SIZE_26
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_25 25
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_26(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_25(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_26(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_25(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_26(_) OF_PP_INTERNAL_SEQ_SIZE_27
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_26 26
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_27(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_26(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_27(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_26(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_27(_) OF_PP_INTERNAL_SEQ_SIZE_28
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_27 27
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_28(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_27(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_28(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_27(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_28(_) OF_PP_INTERNAL_SEQ_SIZE_29
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_28 28
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_29(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_28(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_29(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_28(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_29(_) OF_PP_INTERNAL_SEQ_SIZE_30
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_29 29
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_30(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_29(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_30(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_29(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_30(_) OF_PP_INTERNAL_SEQ_SIZE_31
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_30 30
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_31(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_30(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_31(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_30(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_31(_) OF_PP_INTERNAL_SEQ_SIZE_32
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_31 31
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_32(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_31(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_32(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_31(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_32(_) OF_PP_INTERNAL_SEQ_SIZE_33
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_32 32
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_33(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_32(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_33(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_32(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_33(_) OF_PP_INTERNAL_SEQ_SIZE_34
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_33 33
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_34(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_33(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_34(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_33(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_34(_) OF_PP_INTERNAL_SEQ_SIZE_35
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_34 34
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_35(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_34(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_35(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_34(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_35(_) OF_PP_INTERNAL_SEQ_SIZE_36
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_35 35
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_36(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_35(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_36(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_35(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_36(_) OF_PP_INTERNAL_SEQ_SIZE_37
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_36 36
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_37(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_36(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_37(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_36(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_37(_) OF_PP_INTERNAL_SEQ_SIZE_38
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_37 37
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_38(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_37(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_38(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_37(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_38(_) OF_PP_INTERNAL_SEQ_SIZE_39
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_38 38
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_39(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_38(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_39(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_38(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_39(_) OF_PP_INTERNAL_SEQ_SIZE_40
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_39 39
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_40(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_39(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_40(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_39(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_40(_) OF_PP_INTERNAL_SEQ_SIZE_41
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_40 40
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_41(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_40(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_41(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_40(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_41(_) OF_PP_INTERNAL_SEQ_SIZE_42
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_41 41
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_42(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_41(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_42(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_41(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_42(_) OF_PP_INTERNAL_SEQ_SIZE_43
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_42 42
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_43(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_42(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_43(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_42(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_43(_) OF_PP_INTERNAL_SEQ_SIZE_44
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_43 43
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_44(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_43(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_44(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_43(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_44(_) OF_PP_INTERNAL_SEQ_SIZE_45
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_44 44
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_45(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_44(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_45(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_44(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_45(_) OF_PP_INTERNAL_SEQ_SIZE_46
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_45 45
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_46(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_45(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_46(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_45(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_46(_) OF_PP_INTERNAL_SEQ_SIZE_47
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_46 46
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_47(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_46(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_47(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_46(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_47(_) OF_PP_INTERNAL_SEQ_SIZE_48
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_47 47
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_48(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_47(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_48(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_47(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_48(_) OF_PP_INTERNAL_SEQ_SIZE_49
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_48 48
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_49(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_48(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_49(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_48(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_49(_) OF_PP_INTERNAL_SEQ_SIZE_50
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_49 49
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_50(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_49(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_50(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_49(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_50(_) OF_PP_INTERNAL_SEQ_SIZE_51
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_50 50
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_51(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_50(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_51(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_50(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_51(_) OF_PP_INTERNAL_SEQ_SIZE_52
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_51 51
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_52(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_51(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_52(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_51(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_52(_) OF_PP_INTERNAL_SEQ_SIZE_53
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_52 52
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_53(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_52(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_53(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_52(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_53(_) OF_PP_INTERNAL_SEQ_SIZE_54
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_53 53
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_54(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_53(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_54(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_53(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_54(_) OF_PP_INTERNAL_SEQ_SIZE_55
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_54 54
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_55(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_54(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_55(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_54(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_55(_) OF_PP_INTERNAL_SEQ_SIZE_56
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_55 55
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_56(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_55(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_56(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_55(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_56(_) OF_PP_INTERNAL_SEQ_SIZE_57
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_56 56
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_57(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_56(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_57(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_56(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_57(_) OF_PP_INTERNAL_SEQ_SIZE_58
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_57 57
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_58(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_57(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_58(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_57(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_58(_) OF_PP_INTERNAL_SEQ_SIZE_59
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_58 58
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_59(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_58(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_59(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_58(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_59(_) OF_PP_INTERNAL_SEQ_SIZE_60
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_59 59
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_60(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_59(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_60(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_59(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_60(_) OF_PP_INTERNAL_SEQ_SIZE_61
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_60 60
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_61(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_60(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_61(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_60(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_61(_) OF_PP_INTERNAL_SEQ_SIZE_62
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_61 61
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_62(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_61(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_62(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_61(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_62(_) OF_PP_INTERNAL_SEQ_SIZE_63
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_62 62
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_63(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_62(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_63(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_62(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_63(_) OF_PP_INTERNAL_SEQ_SIZE_64
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_63 63
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_64(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_63(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_64(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_63(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_64(_) OF_PP_INTERNAL_SEQ_SIZE_65
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_64 64
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_65(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_64(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_65(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_64(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_65(_) OF_PP_INTERNAL_SEQ_SIZE_66
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_65 65
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_66(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_65(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_66(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_65(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_66(_) OF_PP_INTERNAL_SEQ_SIZE_67
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_66 66
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_67(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_66(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_67(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_66(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_67(_) OF_PP_INTERNAL_SEQ_SIZE_68
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_67 67
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_68(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_67(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_68(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_67(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_68(_) OF_PP_INTERNAL_SEQ_SIZE_69
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_68 68
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_69(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_68(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_69(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_68(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_69(_) OF_PP_INTERNAL_SEQ_SIZE_70
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_69 69
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_70(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_69(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_70(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_69(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_70(_) OF_PP_INTERNAL_SEQ_SIZE_71
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_70 70
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_71(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_70(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_71(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_70(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_71(_) OF_PP_INTERNAL_SEQ_SIZE_72
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_71 71
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_72(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_71(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_72(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_71(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_72(_) OF_PP_INTERNAL_SEQ_SIZE_73
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_72 72
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_73(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_72(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_73(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_72(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_73(_) OF_PP_INTERNAL_SEQ_SIZE_74
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_73 73
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_74(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_73(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_74(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_73(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_74(_) OF_PP_INTERNAL_SEQ_SIZE_75
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_74 74
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_75(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_74(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_75(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_74(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_75(_) OF_PP_INTERNAL_SEQ_SIZE_76
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_75 75
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_76(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_75(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_76(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_75(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_76(_) OF_PP_INTERNAL_SEQ_SIZE_77
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_76 76
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_77(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_76(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_77(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_76(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_77(_) OF_PP_INTERNAL_SEQ_SIZE_78
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_77 77
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_78(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_77(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_78(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_77(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_78(_) OF_PP_INTERNAL_SEQ_SIZE_79
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_78 78
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_79(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_78(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_79(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_78(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_79(_) OF_PP_INTERNAL_SEQ_SIZE_80
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_79 79
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_80(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_79(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_80(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_79(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_80(_) OF_PP_INTERNAL_SEQ_SIZE_81
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_80 80
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_81(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_80(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_81(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_80(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_81(_) OF_PP_INTERNAL_SEQ_SIZE_82
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_81 81
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_82(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_81(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_82(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_81(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_82(_) OF_PP_INTERNAL_SEQ_SIZE_83
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_82 82
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_83(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_82(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_83(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_82(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_83(_) OF_PP_INTERNAL_SEQ_SIZE_84
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_83 83
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_84(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_83(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_84(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_83(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_84(_) OF_PP_INTERNAL_SEQ_SIZE_85
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_84 84
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_85(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_84(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_85(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_84(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_85(_) OF_PP_INTERNAL_SEQ_SIZE_86
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_85 85
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_86(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_85(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_86(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_85(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_86(_) OF_PP_INTERNAL_SEQ_SIZE_87
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_86 86
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_87(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_86(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_87(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_86(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_87(_) OF_PP_INTERNAL_SEQ_SIZE_88
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_87 87
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_88(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_87(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_88(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_87(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_88(_) OF_PP_INTERNAL_SEQ_SIZE_89
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_88 88
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_89(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_88(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_89(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_88(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_89(_) OF_PP_INTERNAL_SEQ_SIZE_90
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_89 89
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_90(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_89(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_90(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_89(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_90(_) OF_PP_INTERNAL_SEQ_SIZE_91
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_90 90
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_91(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_90(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_91(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_90(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_91(_) OF_PP_INTERNAL_SEQ_SIZE_92
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_91 91
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_92(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_91(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_92(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_91(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_92(_) OF_PP_INTERNAL_SEQ_SIZE_93
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_92 92
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_93(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_92(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_93(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_92(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_93(_) OF_PP_INTERNAL_SEQ_SIZE_94
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_93 93
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_94(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_93(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_94(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_93(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_94(_) OF_PP_INTERNAL_SEQ_SIZE_95
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_94 94
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_95(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_94(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_95(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_94(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_95(_) OF_PP_INTERNAL_SEQ_SIZE_96
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_95 95
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_96(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_95(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_96(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_95(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_96(_) OF_PP_INTERNAL_SEQ_SIZE_97
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_96 96
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_97(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_96(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_97(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_96(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_97(_) OF_PP_INTERNAL_SEQ_SIZE_98
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_97 97
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_98(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_97(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_98(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_97(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_98(_) OF_PP_INTERNAL_SEQ_SIZE_99
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_98 98
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_99(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_98(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_99(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                 \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_98(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_99(_) OF_PP_INTERNAL_SEQ_SIZE_100
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_99 99
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_100(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_99(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_100(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_99(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_100(_) OF_PP_INTERNAL_SEQ_SIZE_101
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_100 100
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_101(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_100(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_101(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_100(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_101(_) OF_PP_INTERNAL_SEQ_SIZE_102
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_101 101
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_102(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_101(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_102(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_101(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_102(_) OF_PP_INTERNAL_SEQ_SIZE_103
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_102 102
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_103(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_102(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_103(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_102(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_103(_) OF_PP_INTERNAL_SEQ_SIZE_104
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_103 103
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_104(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_103(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_104(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_103(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_104(_) OF_PP_INTERNAL_SEQ_SIZE_105
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_104 104
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_105(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_104(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_105(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_104(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_105(_) OF_PP_INTERNAL_SEQ_SIZE_106
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_105 105
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_106(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_105(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_106(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_105(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_106(_) OF_PP_INTERNAL_SEQ_SIZE_107
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_106 106
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_107(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_106(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_107(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_106(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_107(_) OF_PP_INTERNAL_SEQ_SIZE_108
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_107 107
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_108(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_107(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_108(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_107(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_108(_) OF_PP_INTERNAL_SEQ_SIZE_109
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_108 108
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_109(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_108(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_109(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_108(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_109(_) OF_PP_INTERNAL_SEQ_SIZE_110
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_109 109
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_110(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_109(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_110(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_109(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_110(_) OF_PP_INTERNAL_SEQ_SIZE_111
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_110 110
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_111(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_110(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_111(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_110(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_111(_) OF_PP_INTERNAL_SEQ_SIZE_112
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_111 111
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_112(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_111(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_112(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_111(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_112(_) OF_PP_INTERNAL_SEQ_SIZE_113
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_112 112
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_113(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_112(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_113(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_112(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_113(_) OF_PP_INTERNAL_SEQ_SIZE_114
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_113 113
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_114(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_113(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_114(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_113(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_114(_) OF_PP_INTERNAL_SEQ_SIZE_115
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_114 114
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_115(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_114(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_115(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_114(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_115(_) OF_PP_INTERNAL_SEQ_SIZE_116
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_115 115
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_116(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_115(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_116(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_115(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_116(_) OF_PP_INTERNAL_SEQ_SIZE_117
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_116 116
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_117(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_116(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_117(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_116(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_117(_) OF_PP_INTERNAL_SEQ_SIZE_118
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_117 117
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_118(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_117(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_118(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_117(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_118(_) OF_PP_INTERNAL_SEQ_SIZE_119
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_118 118
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_119(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_118(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_119(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_118(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_119(_) OF_PP_INTERNAL_SEQ_SIZE_120
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_119 119
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_120(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_119(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_120(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_119(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_120(_) OF_PP_INTERNAL_SEQ_SIZE_121
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_120 120
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_121(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_120(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_121(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_120(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_121(_) OF_PP_INTERNAL_SEQ_SIZE_122
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_121 121
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_122(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_121(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_122(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_121(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_122(_) OF_PP_INTERNAL_SEQ_SIZE_123
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_122 122
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_123(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_122(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_123(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_122(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_123(_) OF_PP_INTERNAL_SEQ_SIZE_124
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_123 123
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_124(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_123(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_124(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_123(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_124(_) OF_PP_INTERNAL_SEQ_SIZE_125
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_124 124
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_125(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_124(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_125(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_124(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_125(_) OF_PP_INTERNAL_SEQ_SIZE_126
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_125 125
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_126(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_125(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_126(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_125(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_126(_) OF_PP_INTERNAL_SEQ_SIZE_127
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_126 126
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_127(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_126(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_127(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_126(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_127(_) OF_PP_INTERNAL_SEQ_SIZE_128
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_127 127
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_128(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_127(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_128(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_127(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_128(_) OF_PP_INTERNAL_SEQ_SIZE_129
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_128 128
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_129(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_128(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_129(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_128(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_129(_) OF_PP_INTERNAL_SEQ_SIZE_130
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_129 129
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_130(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_129(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_130(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_129(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_130(_) OF_PP_INTERNAL_SEQ_SIZE_131
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_130 130
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_131(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_130(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_131(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_130(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_131(_) OF_PP_INTERNAL_SEQ_SIZE_132
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_131 131
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_132(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_131(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_132(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_131(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_132(_) OF_PP_INTERNAL_SEQ_SIZE_133
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_132 132
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_133(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_132(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_133(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_132(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_133(_) OF_PP_INTERNAL_SEQ_SIZE_134
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_133 133
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_134(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_133(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_134(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_133(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_134(_) OF_PP_INTERNAL_SEQ_SIZE_135
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_134 134
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_135(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_134(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_135(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_134(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_135(_) OF_PP_INTERNAL_SEQ_SIZE_136
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_135 135
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_136(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_135(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_136(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_135(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_136(_) OF_PP_INTERNAL_SEQ_SIZE_137
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_136 136
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_137(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_136(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_137(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_136(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_137(_) OF_PP_INTERNAL_SEQ_SIZE_138
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_137 137
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_138(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_137(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_138(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_137(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_138(_) OF_PP_INTERNAL_SEQ_SIZE_139
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_138 138
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_139(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_138(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_139(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_138(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_139(_) OF_PP_INTERNAL_SEQ_SIZE_140
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_139 139
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_140(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_139(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_140(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_139(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_140(_) OF_PP_INTERNAL_SEQ_SIZE_141
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_140 140
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_141(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_140(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_141(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_140(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_141(_) OF_PP_INTERNAL_SEQ_SIZE_142
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_141 141
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_142(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_141(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_142(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_141(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_142(_) OF_PP_INTERNAL_SEQ_SIZE_143
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_142 142
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_143(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_142(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_143(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_142(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_143(_) OF_PP_INTERNAL_SEQ_SIZE_144
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_143 143
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_144(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_143(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_144(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_143(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_144(_) OF_PP_INTERNAL_SEQ_SIZE_145
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_144 144
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_145(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_144(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_145(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_144(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_145(_) OF_PP_INTERNAL_SEQ_SIZE_146
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_145 145
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_146(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_145(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_146(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_145(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_146(_) OF_PP_INTERNAL_SEQ_SIZE_147
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_146 146
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_147(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_146(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_147(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_146(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_147(_) OF_PP_INTERNAL_SEQ_SIZE_148
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_147 147
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_148(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_147(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_148(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_147(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_148(_) OF_PP_INTERNAL_SEQ_SIZE_149
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_148 148
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_149(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_148(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_149(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_148(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_149(_) OF_PP_INTERNAL_SEQ_SIZE_150
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_149 149
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_150(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_149(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_150(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_149(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_150(_) OF_PP_INTERNAL_SEQ_SIZE_151
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_150 150
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_151(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_150(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_151(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_150(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_151(_) OF_PP_INTERNAL_SEQ_SIZE_152
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_151 151
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_152(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_151(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_152(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_151(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_152(_) OF_PP_INTERNAL_SEQ_SIZE_153
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_152 152
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_153(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_152(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_153(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_152(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_153(_) OF_PP_INTERNAL_SEQ_SIZE_154
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_153 153
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_154(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_153(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_154(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_153(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_154(_) OF_PP_INTERNAL_SEQ_SIZE_155
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_154 154
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_155(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_154(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_155(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_154(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_155(_) OF_PP_INTERNAL_SEQ_SIZE_156
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_155 155
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_156(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_155(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_156(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_155(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_156(_) OF_PP_INTERNAL_SEQ_SIZE_157
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_156 156
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_157(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_156(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_157(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_156(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_157(_) OF_PP_INTERNAL_SEQ_SIZE_158
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_157 157
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_158(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_157(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_158(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_157(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_158(_) OF_PP_INTERNAL_SEQ_SIZE_159
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_158 158
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_159(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_158(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_159(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_158(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_159(_) OF_PP_INTERNAL_SEQ_SIZE_160
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_159 159
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_160(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_159(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_160(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_159(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_160(_) OF_PP_INTERNAL_SEQ_SIZE_161
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_160 160
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_161(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_160(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_161(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_160(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_161(_) OF_PP_INTERNAL_SEQ_SIZE_162
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_161 161
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_162(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_161(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_162(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_161(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_162(_) OF_PP_INTERNAL_SEQ_SIZE_163
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_162 162
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_163(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_162(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_163(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_162(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_163(_) OF_PP_INTERNAL_SEQ_SIZE_164
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_163 163
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_164(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_163(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_164(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_163(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_164(_) OF_PP_INTERNAL_SEQ_SIZE_165
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_164 164
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_165(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_164(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_165(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_164(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_165(_) OF_PP_INTERNAL_SEQ_SIZE_166
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_165 165
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_166(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_165(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_166(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_165(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_166(_) OF_PP_INTERNAL_SEQ_SIZE_167
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_166 166
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_167(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_166(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_167(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_166(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_167(_) OF_PP_INTERNAL_SEQ_SIZE_168
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_167 167
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_168(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_167(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_168(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_167(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_168(_) OF_PP_INTERNAL_SEQ_SIZE_169
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_168 168
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_169(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_168(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_169(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_168(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_169(_) OF_PP_INTERNAL_SEQ_SIZE_170
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_169 169
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_170(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_169(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_170(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_169(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_170(_) OF_PP_INTERNAL_SEQ_SIZE_171
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_170 170
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_171(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_170(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_171(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_170(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_171(_) OF_PP_INTERNAL_SEQ_SIZE_172
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_171 171
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_172(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_171(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_172(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_171(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_172(_) OF_PP_INTERNAL_SEQ_SIZE_173
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_172 172
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_173(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_172(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_173(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_172(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_173(_) OF_PP_INTERNAL_SEQ_SIZE_174
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_173 173
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_174(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_173(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_174(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_173(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_174(_) OF_PP_INTERNAL_SEQ_SIZE_175
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_174 174
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_175(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_174(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_175(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_174(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_175(_) OF_PP_INTERNAL_SEQ_SIZE_176
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_175 175
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_176(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_175(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_176(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_175(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_176(_) OF_PP_INTERNAL_SEQ_SIZE_177
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_176 176
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_177(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_176(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_177(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_176(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_177(_) OF_PP_INTERNAL_SEQ_SIZE_178
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_177 177
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_178(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_177(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_178(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_177(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_178(_) OF_PP_INTERNAL_SEQ_SIZE_179
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_178 178
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_179(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_178(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_179(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_178(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_179(_) OF_PP_INTERNAL_SEQ_SIZE_180
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_179 179
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_180(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_179(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_180(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_179(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_180(_) OF_PP_INTERNAL_SEQ_SIZE_181
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_180 180
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_181(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_180(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_181(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_180(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_181(_) OF_PP_INTERNAL_SEQ_SIZE_182
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_181 181
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_182(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_181(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_182(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_181(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_182(_) OF_PP_INTERNAL_SEQ_SIZE_183
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_182 182
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_183(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_182(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_183(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_182(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_183(_) OF_PP_INTERNAL_SEQ_SIZE_184
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_183 183
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_184(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_183(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_184(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_183(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_184(_) OF_PP_INTERNAL_SEQ_SIZE_185
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_184 184
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_185(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_184(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_185(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_184(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_185(_) OF_PP_INTERNAL_SEQ_SIZE_186
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_185 185
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_186(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_185(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_186(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_185(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_186(_) OF_PP_INTERNAL_SEQ_SIZE_187
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_186 186
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_187(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_186(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_187(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_186(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_187(_) OF_PP_INTERNAL_SEQ_SIZE_188
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_187 187
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_188(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_187(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_188(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_187(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_188(_) OF_PP_INTERNAL_SEQ_SIZE_189
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_188 188
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_189(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_188(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_189(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_188(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_189(_) OF_PP_INTERNAL_SEQ_SIZE_190
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_189 189
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_190(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_189(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_190(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_189(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_190(_) OF_PP_INTERNAL_SEQ_SIZE_191
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_190 190
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_191(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_190(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_191(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_190(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_191(_) OF_PP_INTERNAL_SEQ_SIZE_192
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_191 191
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_192(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_191(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_192(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_191(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_192(_) OF_PP_INTERNAL_SEQ_SIZE_193
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_192 192
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_193(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_192(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_193(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_192(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_193(_) OF_PP_INTERNAL_SEQ_SIZE_194
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_193 193
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_194(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_193(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_194(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_193(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_194(_) OF_PP_INTERNAL_SEQ_SIZE_195
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_194 194
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_195(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_194(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_195(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_194(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_195(_) OF_PP_INTERNAL_SEQ_SIZE_196
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_195 195
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_196(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_195(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_196(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_195(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_196(_) OF_PP_INTERNAL_SEQ_SIZE_197
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_196 196
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_197(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_196(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_197(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_196(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_197(_) OF_PP_INTERNAL_SEQ_SIZE_198
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_197 197
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_198(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_197(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_198(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_197(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_198(_) OF_PP_INTERNAL_SEQ_SIZE_199
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_198 198
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_199(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_198(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_199(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_198(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_199(_) OF_PP_INTERNAL_SEQ_SIZE_200
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_199 199
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_200(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_199(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_200(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_199(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_200(_) OF_PP_INTERNAL_SEQ_SIZE_201
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_200 200
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_201(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_200(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_201(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_200(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_201(_) OF_PP_INTERNAL_SEQ_SIZE_202
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_201 201
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_202(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_201(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_202(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_201(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_202(_) OF_PP_INTERNAL_SEQ_SIZE_203
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_202 202
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_203(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_202(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_203(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_202(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_203(_) OF_PP_INTERNAL_SEQ_SIZE_204
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_203 203
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_204(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_203(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_204(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_203(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_204(_) OF_PP_INTERNAL_SEQ_SIZE_205
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_204 204
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_205(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_204(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_205(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_204(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_205(_) OF_PP_INTERNAL_SEQ_SIZE_206
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_205 205
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_206(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_205(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_206(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_205(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_206(_) OF_PP_INTERNAL_SEQ_SIZE_207
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_206 206
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_207(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_206(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_207(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_206(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_207(_) OF_PP_INTERNAL_SEQ_SIZE_208
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_207 207
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_208(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_207(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_208(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_207(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_208(_) OF_PP_INTERNAL_SEQ_SIZE_209
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_208 208
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_209(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_208(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_209(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_208(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_209(_) OF_PP_INTERNAL_SEQ_SIZE_210
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_209 209
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_210(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_209(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_210(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_209(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_210(_) OF_PP_INTERNAL_SEQ_SIZE_211
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_210 210
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_211(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_210(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_211(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_210(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_211(_) OF_PP_INTERNAL_SEQ_SIZE_212
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_211 211
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_212(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_211(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_212(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_211(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_212(_) OF_PP_INTERNAL_SEQ_SIZE_213
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_212 212
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_213(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_212(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_213(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_212(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_213(_) OF_PP_INTERNAL_SEQ_SIZE_214
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_213 213
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_214(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_213(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_214(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_213(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_214(_) OF_PP_INTERNAL_SEQ_SIZE_215
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_214 214
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_215(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_214(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_215(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_214(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_215(_) OF_PP_INTERNAL_SEQ_SIZE_216
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_215 215
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_216(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_215(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_216(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_215(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_216(_) OF_PP_INTERNAL_SEQ_SIZE_217
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_216 216
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_217(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_216(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_217(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_216(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_217(_) OF_PP_INTERNAL_SEQ_SIZE_218
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_217 217
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_218(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_217(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_218(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_217(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_218(_) OF_PP_INTERNAL_SEQ_SIZE_219
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_218 218
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_219(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_218(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_219(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_218(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_219(_) OF_PP_INTERNAL_SEQ_SIZE_220
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_219 219
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_220(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_219(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_220(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_219(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_220(_) OF_PP_INTERNAL_SEQ_SIZE_221
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_220 220
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_221(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_220(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_221(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_220(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_221(_) OF_PP_INTERNAL_SEQ_SIZE_222
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_221 221
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_222(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_221(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_222(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_221(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_222(_) OF_PP_INTERNAL_SEQ_SIZE_223
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_222 222
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_223(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_222(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_223(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_222(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_223(_) OF_PP_INTERNAL_SEQ_SIZE_224
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_223 223
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_224(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_223(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_224(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_223(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_224(_) OF_PP_INTERNAL_SEQ_SIZE_225
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_224 224
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_225(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_224(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_225(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_224(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_225(_) OF_PP_INTERNAL_SEQ_SIZE_226
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_225 225
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_226(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_225(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_226(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_225(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_226(_) OF_PP_INTERNAL_SEQ_SIZE_227
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_226 226
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_227(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_226(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_227(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_226(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_227(_) OF_PP_INTERNAL_SEQ_SIZE_228
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_227 227
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_228(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_227(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_228(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_227(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_228(_) OF_PP_INTERNAL_SEQ_SIZE_229
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_228 228
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_229(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_228(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_229(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_228(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_229(_) OF_PP_INTERNAL_SEQ_SIZE_230
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_229 229
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_230(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_229(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_230(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_229(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_230(_) OF_PP_INTERNAL_SEQ_SIZE_231
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_230 230
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_231(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_230(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_231(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_230(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_231(_) OF_PP_INTERNAL_SEQ_SIZE_232
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_231 231
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_232(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_231(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_232(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_231(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_232(_) OF_PP_INTERNAL_SEQ_SIZE_233
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_232 232
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_233(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_232(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_233(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_232(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_233(_) OF_PP_INTERNAL_SEQ_SIZE_234
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_233 233
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_234(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_233(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_234(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_233(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_234(_) OF_PP_INTERNAL_SEQ_SIZE_235
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_234 234
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_235(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_234(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_235(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_234(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_235(_) OF_PP_INTERNAL_SEQ_SIZE_236
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_235 235
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_236(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_235(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_236(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_235(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_236(_) OF_PP_INTERNAL_SEQ_SIZE_237
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_236 236
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_237(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_236(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_237(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_236(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_237(_) OF_PP_INTERNAL_SEQ_SIZE_238
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_237 237
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_238(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_237(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_238(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_237(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_238(_) OF_PP_INTERNAL_SEQ_SIZE_239
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_238 238
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_239(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_238(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_239(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_238(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_239(_) OF_PP_INTERNAL_SEQ_SIZE_240
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_239 239
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_240(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_239(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_240(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_239(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_240(_) OF_PP_INTERNAL_SEQ_SIZE_241
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_240 240
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_241(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_240(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_241(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_240(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_241(_) OF_PP_INTERNAL_SEQ_SIZE_242
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_241 241
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_242(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_241(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_242(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_241(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_242(_) OF_PP_INTERNAL_SEQ_SIZE_243
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_242 242
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_243(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_242(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_243(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_242(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_243(_) OF_PP_INTERNAL_SEQ_SIZE_244
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_243 243
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_244(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_243(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_244(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_243(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_244(_) OF_PP_INTERNAL_SEQ_SIZE_245
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_244 244
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_245(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_244(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_245(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_244(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_245(_) OF_PP_INTERNAL_SEQ_SIZE_246
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_245 245
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_246(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_245(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_246(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_245(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_246(_) OF_PP_INTERNAL_SEQ_SIZE_247
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_246 246
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_247(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_246(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_247(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_246(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_247(_) OF_PP_INTERNAL_SEQ_SIZE_248
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_247 247
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_248(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_247(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_248(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_247(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_248(_) OF_PP_INTERNAL_SEQ_SIZE_249
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_248 248
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_249(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_248(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_249(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_248(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_249(_) OF_PP_INTERNAL_SEQ_SIZE_250
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_249 249
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_250(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_249(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_250(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_249(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_250(_) OF_PP_INTERNAL_SEQ_SIZE_251
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_250 250
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_251(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_250(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_251(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_250(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_251(_) OF_PP_INTERNAL_SEQ_SIZE_252
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_251 251
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_252(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_251(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_252(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_251(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_252(_) OF_PP_INTERNAL_SEQ_SIZE_253
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_252 252
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_253(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_252(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_253(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_252(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_253(_) OF_PP_INTERNAL_SEQ_SIZE_254
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_253 253
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_254(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_253(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_254(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_253(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_254(_) OF_PP_INTERNAL_SEQ_SIZE_255
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_254 254
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_255(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_254(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_255(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_254(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#define OF_PP_INTERNAL_SEQ_SIZE_255(_) OF_PP_INTERNAL_SEQ_SIZE_256
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_255 255
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_256(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D1_SEQ_FOR_EACH_255(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_256(apply, m, d, seq) \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq))                  \
      OF_PP_INTERNAL_D2_SEQ_FOR_EACH_255(apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#endif  // ONEFLOW_CORE_COMMON_PREPROCESSOR_INTERNAL_H_
