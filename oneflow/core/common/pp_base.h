#ifndef ONEFLOW_CORE_COMMON_PP_BASE_H_
#define ONEFLOW_CORE_COMMON_PP_BASE_H_

#define OF_PP_STRINGIZE(text) OF_PP_STRINGIZE_I(text)
#define OF_PP_STRINGIZE_I(text) #text

#define OF_PP_CAT(a, b) OF_PP_CAT_I(a, b)
#define OF_PP_CAT_I(a, b) a##b

#define OF_PP_SEQ_HEAD(seq) OF_PP_PAIR_FIRST(OF_PP_SEQ_TO_PAIR(seq))
#define OF_PP_SEQ_TAIL(seq) OF_PP_PAIR_SECOND(OF_PP_SEQ_TO_PAIR(seq))

#define OF_PP_SEQ_TO_PAIR(seq) (OF_PP_SEQ_TO_PAIR_ seq)
#define OF_PP_SEQ_TO_PAIR_(x) x, OF_PP_NIL
#define OF_PP_NIL

#define OF_PP_PAIR_FIRST(t) OF_PP_PAIR_FIRST_I(t)
#define OF_PP_PAIR_FIRST_I(t) OF_PP_FIRST_ARG t

#define OF_PP_PAIR_SECOND(t) OF_PP_PAIR_SECOND_I(t)
#define OF_PP_PAIR_SECOND_I(t) OF_PP_SECOND_ARG t

#define OF_PP_FIRST_ARG(x, ...) x
#define OF_PP_SECOND_ARG(x, y, ...) y

#define OF_PP_MAKE_TUPLE(...) (__VA_ARGS__)
#define OF_PP_MAKE_TUPLE_SEQ(...) (OF_PP_MAKE_TUPLE(__VA_ARGS__))

#endif  // ONEFLOW_CORE_COMMON_PP_BASE_H_
