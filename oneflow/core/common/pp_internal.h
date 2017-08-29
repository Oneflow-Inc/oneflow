#ifndef ONEFLOW_CORE_COMMON_PP_INTERNAL_H_
#define ONEFLOW_CORE_COMMON_PP_INTERNAL_H_

// seq example0: (0)(1)(2)
// seq example1: (0)(1)(2)

//	pp base
#define OF_PP_INTERNAL_STRINGIZE(text) OF_PP_INTERNAL_STRINGIZE_I(text)
#define OF_PP_INTERNAL_STRINGIZE_I(text) #text

#define OF_PP_INTERNAL_CAT(a, b) OF_PP_INTERNAL_CAT_I(a, b)
#define OF_PP_INTERNAL_CAT_I(a, b) a##b

#define OF_PP_INTERNAL_SEQ_HEAD(seq) \
  OF_PP_INTERNAL_PAIR_FIRST(OF_PP_INTERNAL_SEQ_TO_PAIR(seq))
#define OF_PP_INTERNAL_SEQ_TAIL(seq) \
  OF_PP_INTERNAL_PAIR_SECOND(OF_PP_INTERNAL_SEQ_TO_PAIR(seq))

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
#define OF_PP_INTERNAL_MAKE_TUPLE_SEQ(...) \
  (OF_PP_INTERNAL_MAKE_TUPLE(__VA_ARGS__))

//	tuple
#define OF_PP_INTERNAL_TUPLE_PUSH_FRONT(tuple, x)          \
  OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_TUPLE_PUSH_FRONT_,     \
                     OF_PP_INTERNAL_IS_TUPLE_EMPTY(tuple)) \
  (tuple, x)

#define OF_PP_INTERNAL_TUPLE_PUSH_FRONT_1(tuple, x) (x)
#define OF_PP_INTERNAL_TUPLE_PUSH_FRONT_0(tuple, x) \
  (x, OF_PP_INTERNAL_TUPLE_TO_ARGS(tuple))

#define OF_PP_INTERNAL_TUPLE_TO_ARGS(t) OF_PP_INTERNAL_TUPLE_TO_ARGS_I(t)
#define OF_PP_INTERNAL_TUPLE_TO_ARGS_I(t) OF_PP_INTERNAL_TUPLE_TO_ARGS_ t
#define OF_PP_INTERNAL_TUPLE_TO_ARGS_(...) __VA_ARGS__

#define OF_PP_INTERNAL_TUPLE_SIZE(tuple)                   \
  OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_TUPLE_SIZE_,           \
                     OF_PP_INTERNAL_IS_TUPLE_EMPTY(tuple)) \
  (tuple)

#define OF_PP_INTERNAL_TUPLE_SIZE_1(t) 0
#define OF_PP_INTERNAL_TUPLE_SIZE_0(t) OF_PP_INTERNAL_TUPLE_SIZE_0_I(t)
#define OF_PP_INTERNAL_TUPLE_SIZE_0_I(t) OF_PP_INTERNAL_VARIADIC_SIZE t

#define OF_PP_INTERNAL_VARIADIC_SIZE(...)                                      \
  OF_PP_INTERNAL_VARIADIC_SIZE_I(                                              \
      __VA_ARGS__, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, \
      49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32,  \
      31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14,  \
      13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, )
#define OF_PP_INTERNAL_VARIADIC_SIZE_I(                                        \
    e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, \
    e17, e18, e19, e20, e21, e22, e23, e24, e25, e26, e27, e28, e29, e30, e31, \
    e32, e33, e34, e35, e36, e37, e38, e39, e40, e41, e42, e43, e44, e45, e46, \
    e47, e48, e49, e50, e51, e52, e53, e54, e55, e56, e57, e58, e59, e60, e61, \
    e62, e63, size, ...)                                                       \
  size

#define OF_PP_INTERNAL_IS_TUPLE_EMPTY(t) \
  OF_PP_INTERNAL_IS_VARIADIC_EMPTY(OF_PP_INTERNAL_TUPLE_TO_ARGS(t))

#define OF_PP_INTERNAL_IS_VARIADIC_EMPTY(...)                                                    \
  OF_PP_INTERNAL_IS_VARIADIC_EMPTY_(/* test if there is just one argument,                       \
                              eventually an empty one */                                         \
                                    OF_PP_INTERNAL_VARIADIC_HAS_COMMA(                           \
                                        __VA_ARGS__), /* test if                                 \
                                                         _OF_PP_INTERNAL_TRIGGER_PARENTHESIS_    \
                                                         together with the                       \
                                                         argument adds a comma                   \
                                                       */                                        \
                                    OF_PP_INTERNAL_VARIADIC_HAS_COMMA(                           \
                                        _OF_PP_INTERNAL_TRIGGER_PARENTHESIS_                     \
                                            __VA_ARGS__), /* test if the                         \
                                                             argument together                   \
                                                             with a                              \
                                                             parenthesis adds                    \
                                                             a comma                             \
                                                           */                                    \
                                    OF_PP_INTERNAL_VARIADIC_HAS_COMMA(                           \
                                        __VA_ARGS__(                                             \
                                            /*empty*/)), /* test if placing it                   \
                                                            between                              \
                                                            _OF_PP_INTERNAL_TRIGGER_PARENTHESIS_ \
                                                            and the                              \
                                                            parenthesis adds a                   \
                                                            comma */                             \
                                    OF_PP_INTERNAL_VARIADIC_HAS_COMMA(                           \
                                        _OF_PP_INTERNAL_TRIGGER_PARENTHESIS_                     \
                                            __VA_ARGS__(/*empty*/)))

#define OF_PP_INTERNAL_IS_VARIADIC_EMPTY_(e0, e1, e2, e3) \
  OF_PP_INTERNAL_VARIADIC_HAS_COMMA(                      \
      OF_PP_INTERNAL_CAT5(OF_PP_INTERNAL_IS_EMPTY_CASE_, e0, e1, e2, e3))

#define OF_PP_INTERNAL_VARIADIC_HAS_COMMA(...)                                 \
  OF_PP_INTERNAL_VARIADIC_HAS_COMMA_I(                                         \
      __VA_ARGS__, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  \
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0)
#define OF_PP_INTERNAL_VARIADIC_HAS_COMMA_I(                                   \
    e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, \
    e17, e18, e19, e20, e21, e22, e23, e24, e25, e26, e27, e28, e29, e30, e31, \
    e32, e33, e34, e35, e36, e37, e38, e39, e40, e41, e42, e43, e44, e45, e46, \
    e47, e48, e49, e50, e51, e52, e53, e54, e55, e56, e57, e58, e59, e60, e61, \
    e62, e63, has_comma, ...)                                                  \
  has_comma

#define _OF_PP_INTERNAL_TRIGGER_PARENTHESIS_(...) ,

#define OF_PP_INTERNAL_CAT5(e0, e1, e2, e3, e4) e0##e1##e2##e3##e4
#define OF_PP_INTERNAL_IS_EMPTY_CASE_0001 ,

//	pp seq product

#define OF_PP_INTERNAL_SEQ_PRODUCT_FOR_EACH_TUPLE(macro, ...) \
  OF_PP_INTERNAL_SEQ_FOR_EACH_TUPLE(macro, _,                 \
                                    OF_PP_INTERNAL_SEQ_PRODUCT(__VA_ARGS__))

#define OF_PP_INTERNAL_SEQ_PRODUCT(...)                        \
  OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_SEQ_PRODUCT_,              \
                     OF_PP_INTERNAL_TUPLE_SIZE((__VA_ARGS__))) \
  (__VA_ARGS__)

#define OF_PP_INTERNAL_SEQ_PRODUCT_0()
#define OF_PP_INTERNAL_SEQ_PRODUCT_1(seq0) \
  OF_PP_INTERNAL_TUPLE_SEQ_X_ATOMIC_SEQ((()), seq0)
#define OF_PP_INTERNAL_SEQ_PRODUCT_2(seq0, seq1)                            \
  OF_PP_INTERNAL_TUPLE_SEQ_X_ATOMIC_SEQ(OF_PP_INTERNAL_SEQ_PRODUCT_1(seq1), \
                                        seq0)
#define OF_PP_INTERNAL_SEQ_PRODUCT_3(seq0, seq1, seq2) \
  OF_PP_INTERNAL_TUPLE_SEQ_X_ATOMIC_SEQ(               \
      OF_PP_INTERNAL_SEQ_PRODUCT_2(seq1, seq2), seq0)
#define OF_PP_INTERNAL_SEQ_PRODUCT_4(seq0, seq1, seq2, seq3) \
  OF_PP_INTERNAL_TUPLE_SEQ_X_ATOMIC_SEQ(                     \
      OF_PP_INTERNAL_SEQ_PRODUCT_3(seq1, seq2, seq3), seq0)
#define OF_PP_INTERNAL_SEQ_PRODUCT_5(seq0, seq1, seq2, seq3, seq4) \
  OF_PP_INTERNAL_TUPLE_SEQ_X_ATOMIC_SEQ(                           \
      OF_PP_INTERNAL_SEQ_PRODUCT_4(seq1, seq2, seq3, seq4), seq0)

//	pp seq for each
#define OF_PP_INTERNAL_FOR_EACH_TUPLE(macro, seq) \
  OF_PP_INTERNAL_SEQ_FOR_EACH_TUPLE(macro, _, seq)
#define OF_PP_INTERNAL_TUPLE_SEQ_X_ATOMIC_SEQ(tuple_seq, atomic_seq)       \
  OF_PP_INTERNAL_D1_SEQ_FOR_EACH(OF_PP_INTERNAL_D1_APPLY_ATOMIC_WITH_DATA, \
                                 OF_PP_INTERNAL_TUPLE_X_ATOMIC_SEQ,        \
                                 atomic_seq, tuple_seq)

#define OF_PP_INTERNAL_TUPLE_X_ATOMIC_SEQ(atomic_seq, tuple)               \
  OF_PP_INTERNAL_D2_SEQ_FOR_EACH(OF_PP_INTERNAL_D2_APPLY_ATOMIC_WITH_DATA, \
                                 OF_PP_INTERNAL_MAKE_SEQ_TUPLE_PUSH_FRONT, \
                                 tuple, atomic_seq)

#define OF_PP_INTERNAL_D1_APPLY_ATOMIC_WITH_DATA(m, d, x) m(d, x)
#define OF_PP_INTERNAL_D2_APPLY_ATOMIC_WITH_DATA(m, d, x) m(d, x)

#define OF_PP_INTERNAL_MAKE_SEQ_TUPLE_PUSH_FRONT(tuple, x) \
  (OF_PP_INTERNAL_TUPLE_PUSH_FRONT(tuple, x))

//	pp seq size
#define OF_PP_INTERNAL_SEQ_SIZE(seq) OF_PP_INTERNAL_SEQ_SIZE_I(seq)
#define OF_PP_INTERNAL_SEQ_SIZE_I(seq) \
  OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_SEQ_SIZE_, OF_PP_INTERNAL_SEQ_SIZE_0 seq)
#define OF_PP_INTERNAL_SEQ_SIZE_0(_) OF_PP_INTERNAL_SEQ_SIZE_1
#define OF_PP_INTERNAL_SEQ_SIZE_1(_) OF_PP_INTERNAL_SEQ_SIZE_2
#define OF_PP_INTERNAL_SEQ_SIZE_2(_) OF_PP_INTERNAL_SEQ_SIZE_3
#define OF_PP_INTERNAL_SEQ_SIZE_3(_) OF_PP_INTERNAL_SEQ_SIZE_4
#define OF_PP_INTERNAL_SEQ_SIZE_4(_) OF_PP_INTERNAL_SEQ_SIZE_5
#define OF_PP_INTERNAL_SEQ_SIZE_5(_) OF_PP_INTERNAL_SEQ_SIZE_6
#define OF_PP_INTERNAL_SEQ_SIZE_6(_) OF_PP_INTERNAL_SEQ_SIZE_7
#define OF_PP_INTERNAL_SEQ_SIZE_7(_) OF_PP_INTERNAL_SEQ_SIZE_8
#define OF_PP_INTERNAL_SEQ_SIZE_8(_) OF_PP_INTERNAL_SEQ_SIZE_9
#define OF_PP_INTERNAL_SEQ_SIZE_9(_) OF_PP_INTERNAL_SEQ_SIZE_10
#define OF_PP_INTERNAL_SEQ_SIZE_10(_) OF_PP_INTERNAL_SEQ_SIZE_11
#define OF_PP_INTERNAL_SEQ_SIZE_11(_) OF_PP_INTERNAL_SEQ_SIZE_12
#define OF_PP_INTERNAL_SEQ_SIZE_12(_) OF_PP_INTERNAL_SEQ_SIZE_13
#define OF_PP_INTERNAL_SEQ_SIZE_13(_) OF_PP_INTERNAL_SEQ_SIZE_14
#define OF_PP_INTERNAL_SEQ_SIZE_14(_) OF_PP_INTERNAL_SEQ_SIZE_15
#define OF_PP_INTERNAL_SEQ_SIZE_15(_) OF_PP_INTERNAL_SEQ_SIZE_16
#define OF_PP_INTERNAL_SEQ_SIZE_16(_) OF_PP_INTERNAL_SEQ_SIZE_17
#define OF_PP_INTERNAL_SEQ_SIZE_17(_) OF_PP_INTERNAL_SEQ_SIZE_18
#define OF_PP_INTERNAL_SEQ_SIZE_18(_) OF_PP_INTERNAL_SEQ_SIZE_19
#define OF_PP_INTERNAL_SEQ_SIZE_19(_) OF_PP_INTERNAL_SEQ_SIZE_20
#define OF_PP_INTERNAL_SEQ_SIZE_20(_) OF_PP_INTERNAL_SEQ_SIZE_21
#define OF_PP_INTERNAL_SEQ_SIZE_21(_) OF_PP_INTERNAL_SEQ_SIZE_22
#define OF_PP_INTERNAL_SEQ_SIZE_22(_) OF_PP_INTERNAL_SEQ_SIZE_23
#define OF_PP_INTERNAL_SEQ_SIZE_23(_) OF_PP_INTERNAL_SEQ_SIZE_24
#define OF_PP_INTERNAL_SEQ_SIZE_24(_) OF_PP_INTERNAL_SEQ_SIZE_25
#define OF_PP_INTERNAL_SEQ_SIZE_25(_) OF_PP_INTERNAL_SEQ_SIZE_26
#define OF_PP_INTERNAL_SEQ_SIZE_26(_) OF_PP_INTERNAL_SEQ_SIZE_27
#define OF_PP_INTERNAL_SEQ_SIZE_27(_) OF_PP_INTERNAL_SEQ_SIZE_28
#define OF_PP_INTERNAL_SEQ_SIZE_28(_) OF_PP_INTERNAL_SEQ_SIZE_29
#define OF_PP_INTERNAL_SEQ_SIZE_29(_) OF_PP_INTERNAL_SEQ_SIZE_30
#define OF_PP_INTERNAL_SEQ_SIZE_30(_) OF_PP_INTERNAL_SEQ_SIZE_31
#define OF_PP_INTERNAL_SEQ_SIZE_31(_) OF_PP_INTERNAL_SEQ_SIZE_32
#define OF_PP_INTERNAL_SEQ_SIZE_32(_) OF_PP_INTERNAL_SEQ_SIZE_33
#define OF_PP_INTERNAL_SEQ_SIZE_33(_) OF_PP_INTERNAL_SEQ_SIZE_34
#define OF_PP_INTERNAL_SEQ_SIZE_34(_) OF_PP_INTERNAL_SEQ_SIZE_35
#define OF_PP_INTERNAL_SEQ_SIZE_35(_) OF_PP_INTERNAL_SEQ_SIZE_36
#define OF_PP_INTERNAL_SEQ_SIZE_36(_) OF_PP_INTERNAL_SEQ_SIZE_37
#define OF_PP_INTERNAL_SEQ_SIZE_37(_) OF_PP_INTERNAL_SEQ_SIZE_38
#define OF_PP_INTERNAL_SEQ_SIZE_38(_) OF_PP_INTERNAL_SEQ_SIZE_39
#define OF_PP_INTERNAL_SEQ_SIZE_39(_) OF_PP_INTERNAL_SEQ_SIZE_40
#define OF_PP_INTERNAL_SEQ_SIZE_40(_) OF_PP_INTERNAL_SEQ_SIZE_41
#define OF_PP_INTERNAL_SEQ_SIZE_41(_) OF_PP_INTERNAL_SEQ_SIZE_42
#define OF_PP_INTERNAL_SEQ_SIZE_42(_) OF_PP_INTERNAL_SEQ_SIZE_43
#define OF_PP_INTERNAL_SEQ_SIZE_43(_) OF_PP_INTERNAL_SEQ_SIZE_44
#define OF_PP_INTERNAL_SEQ_SIZE_44(_) OF_PP_INTERNAL_SEQ_SIZE_45
#define OF_PP_INTERNAL_SEQ_SIZE_45(_) OF_PP_INTERNAL_SEQ_SIZE_46
#define OF_PP_INTERNAL_SEQ_SIZE_46(_) OF_PP_INTERNAL_SEQ_SIZE_47
#define OF_PP_INTERNAL_SEQ_SIZE_47(_) OF_PP_INTERNAL_SEQ_SIZE_48
#define OF_PP_INTERNAL_SEQ_SIZE_48(_) OF_PP_INTERNAL_SEQ_SIZE_49
#define OF_PP_INTERNAL_SEQ_SIZE_49(_) OF_PP_INTERNAL_SEQ_SIZE_50
#define OF_PP_INTERNAL_SEQ_SIZE_50(_) OF_PP_INTERNAL_SEQ_SIZE_51
#define OF_PP_INTERNAL_SEQ_SIZE_51(_) OF_PP_INTERNAL_SEQ_SIZE_52
#define OF_PP_INTERNAL_SEQ_SIZE_52(_) OF_PP_INTERNAL_SEQ_SIZE_53
#define OF_PP_INTERNAL_SEQ_SIZE_53(_) OF_PP_INTERNAL_SEQ_SIZE_54
#define OF_PP_INTERNAL_SEQ_SIZE_54(_) OF_PP_INTERNAL_SEQ_SIZE_55
#define OF_PP_INTERNAL_SEQ_SIZE_55(_) OF_PP_INTERNAL_SEQ_SIZE_56
#define OF_PP_INTERNAL_SEQ_SIZE_56(_) OF_PP_INTERNAL_SEQ_SIZE_57
#define OF_PP_INTERNAL_SEQ_SIZE_57(_) OF_PP_INTERNAL_SEQ_SIZE_58
#define OF_PP_INTERNAL_SEQ_SIZE_58(_) OF_PP_INTERNAL_SEQ_SIZE_59
#define OF_PP_INTERNAL_SEQ_SIZE_59(_) OF_PP_INTERNAL_SEQ_SIZE_60
#define OF_PP_INTERNAL_SEQ_SIZE_60(_) OF_PP_INTERNAL_SEQ_SIZE_61
#define OF_PP_INTERNAL_SEQ_SIZE_61(_) OF_PP_INTERNAL_SEQ_SIZE_62
#define OF_PP_INTERNAL_SEQ_SIZE_62(_) OF_PP_INTERNAL_SEQ_SIZE_63
#define OF_PP_INTERNAL_SEQ_SIZE_63(_) OF_PP_INTERNAL_SEQ_SIZE_64
#define OF_PP_INTERNAL_SEQ_SIZE_64(_) OF_PP_INTERNAL_SEQ_SIZE_65

#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_0 0
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_1 1
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_2 2
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_3 3
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_4 4
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_5 5
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_6 6
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_7 7
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_8 8
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_9 9
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_10 10
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_11 11
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_12 12
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_13 13
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_14 14
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_15 15
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_16 16
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_17 17
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_18 18
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_19 19
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_20 20
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_21 21
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_22 22
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_23 23
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_24 24
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_25 25
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_26 26
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_27 27
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_28 28
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_29 29
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_30 30
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_31 31
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_32 32
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_33 33
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_34 34
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_35 35
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_36 36
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_37 37
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_38 38
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_39 39
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_40 40
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_41 41
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_42 42
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_43 43
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_44 44
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_45 45
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_46 46
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_47 47
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_48 48
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_49 49
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_50 50
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_51 51
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_52 52
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_53 53
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_54 54
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_55 55
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_56 56
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_57 57
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_58 58
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_59 59
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_60 60
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_61 61
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_62 62
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_63 63
#define OF_PP_INTERNAL_SEQ_SIZE_OF_PP_INTERNAL_SEQ_SIZE_64 64

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

#define OF_PP_INTERNAL_APPLY_TUPLE(m, d, t) \
  OF_PP_INTERNAL_APPLY_TUPLE_I(m, d, t)
#define OF_PP_INTERNAL_APPLY_TUPLE_I(m, d, t) m t
#define OF_PP_INTERNAL_APPLY_ATOMIC(m, d, x) m(x)
#define OF_PP_INTERNAL_APPLY_ATOMIC_WITH_DATA(m, d, x) m(d, x)

#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH(apply, m, d, seq) \
  OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_D1_SEQ_FOR_EACH_,    \
                     OF_PP_INTERNAL_SEQ_SIZE(seq))       \
  (apply, m, d, seq)

#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH(apply, m, d, seq) \
  OF_PP_INTERNAL_CAT(OF_PP_INTERNAL_D2_SEQ_FOR_EACH_,    \
                     OF_PP_INTERNAL_SEQ_SIZE(seq))       \
  (apply, m, d, seq)

#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_0(apply, m, d, seq)
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_0(apply, m, d, seq)

#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_1(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_0( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_1(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_0( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_2(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_1( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_2(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_1( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_3(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_2( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_3(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_2( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_4(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_3( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_4(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_3( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_5(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_4( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_5(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_4( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_6(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_5( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_6(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_5( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_7(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_6( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_7(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_6( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_8(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_7( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_8(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_7( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_9(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_8( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_9(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_8( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_10(apply, m, d, seq)                   \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_9( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_10(apply, m, d, seq)                   \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_9( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_11(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_10( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_11(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_10( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_12(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_11( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_12(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_11( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_13(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_12( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_13(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_12( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_14(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_13( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_14(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_13( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_15(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_14( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_15(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_14( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_16(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_15( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_16(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_15( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_17(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_16( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_17(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_16( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_18(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_17( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_18(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_17( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_19(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_18( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_19(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_18( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_20(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_19( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_20(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_19( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_21(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_20( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_21(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_20( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_22(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_21( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_22(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_21( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_23(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_22( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_23(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_22( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_24(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_23( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_24(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_23( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_25(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_24( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_25(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_24( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_26(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_25( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_26(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_25( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_27(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_26( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_27(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_26( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_28(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_27( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_28(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_27( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_29(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_28( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_29(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_28( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_30(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_29( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_30(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_29( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_31(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_30( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_31(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_30( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_32(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_31( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_32(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_31( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_33(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_32( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_33(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_32( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_34(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_33( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_34(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_33( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_35(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_34( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_35(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_34( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_36(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_35( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_36(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_35( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_37(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_36( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_37(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_36( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_38(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_37( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_38(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_37( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_39(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_38( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_39(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_38( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_40(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_39( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_40(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_39( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_41(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_40( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_41(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_40( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_42(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_41( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_42(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_41( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_43(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_42( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_43(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_42( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_44(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_43( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_44(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_43( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_45(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_44( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_45(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_44( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_46(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_45( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_46(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_45( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_47(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_46( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_47(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_46( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_48(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_47( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_48(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_47( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_49(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_48( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_49(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_48( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_50(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_49( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_50(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_49( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_51(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_50( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_51(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_50( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_52(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_51( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_52(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_51( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_53(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_52( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_53(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_52( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_54(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_53( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_54(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_53( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_55(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_54( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_55(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_54( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_56(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_55( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_56(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_55( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_57(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_56( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_57(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_56( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_58(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_57( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_58(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_57( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_59(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_58( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_59(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_58( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_60(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_59( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_60(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_59( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_61(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_60( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_61(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_60( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_62(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_61( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_62(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_61( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_63(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_62( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_63(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_62( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_64(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_63( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_64(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_63( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D1_SEQ_FOR_EACH_65(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D1_SEQ_FOR_EACH_64( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))
#define OF_PP_INTERNAL_D2_SEQ_FOR_EACH_65(apply, m, d, seq)                    \
  apply(m, d, OF_PP_INTERNAL_SEQ_HEAD(seq)) OF_PP_INTERNAL_D2_SEQ_FOR_EACH_64( \
      apply, m, d, OF_PP_INTERNAL_SEQ_TAIL(seq))

#endif  // ONEFLOW_CORE_COMMON_PP_INTERNAL_H_
