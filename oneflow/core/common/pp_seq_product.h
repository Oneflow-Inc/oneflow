#ifndef ONEFLOW_CORE_COMMON_PP_SEQ_PRODUCT_
#define ONEFLOW_CORE_COMMON_PP_SEQ_PRODUCT_

#include "oneflow/core/common/pp_seq_for_each.h"
#include "oneflow/core/common/pp_tuple.h"

#define OF_PP_TUPLE_SEQ_X_ATOMIC_SEQ(tuple_seq, atomic_seq) \
  OF_PP_D1_SEQ_FOR_EACH(OF_PP_D1_APPLY_ATOMIC_WITH_DATA,    \
                        OF_PP_TUPLE_X_ATOMIC_SEQ, atomic_seq, tuple_seq)

#define OF_PP_TUPLE_X_ATOMIC_SEQ(atomic_seq, tuple)      \
  OF_PP_D2_SEQ_FOR_EACH(OF_PP_D2_APPLY_ATOMIC_WITH_DATA, \
                        OF_PP_MAKE_SEQ_TUPLE_PUSH_FRONT, tuple, atomic_seq)

#define OF_PP_D1_APPLY_ATOMIC_WITH_DATA(m, d, x) m(d, x)
#define OF_PP_D2_APPLY_ATOMIC_WITH_DATA(m, d, x) m(d, x)

#define OF_PP_MAKE_SEQ_TUPLE_PUSH_FRONT(tuple, x) \
  (OF_PP_TUPLE_PUSH_FRONT(tuple, x))

#define OF_PP_SEQ_PRODUCT(...) \
  OF_PP_CAT(OF_PP_SEQ_PRODUCT_, OF_PP_TUPLE_SIZE((__VA_ARGS__)))(__VA_ARGS__)

#define OF_PP_SEQ_PRODUCT_0() (())
#define OF_PP_SEQ_PRODUCT_1(seq0) \
  OF_PP_TUPLE_SEQ_X_ATOMIC_SEQ(OF_PP_SEQ_PRODUCT_0(), seq0)
#define OF_PP_SEQ_PRODUCT_2(seq0, seq1) \
  OF_PP_TUPLE_SEQ_X_ATOMIC_SEQ(OF_PP_SEQ_PRODUCT_1(seq1), seq0)
#define OF_PP_SEQ_PRODUCT_3(seq0, seq1, seq2) \
  OF_PP_TUPLE_SEQ_X_ATOMIC_SEQ(OF_PP_SEQ_PRODUCT_2(seq1, seq2), seq0)
#define OF_PP_SEQ_PRODUCT_4(seq0, seq1, seq2, seq3) \
  OF_PP_TUPLE_SEQ_X_ATOMIC_SEQ(OF_PP_SEQ_PRODUCT_3(seq1, seq2, seq3), seq0)
#define OF_PP_SEQ_PRODUCT_5(seq0, seq1, seq2, seq3, seq4)                   \
  OF_PP_TUPLE_SEQ_X_ATOMIC_SEQ(OF_PP_SEQ_PRODUCT_4(seq1, seq2, seq3, seq4), \
                               seq0)
#endif  // ONEFLOW_CORE_COMMON_PP_SEQ_PRODUCT_
