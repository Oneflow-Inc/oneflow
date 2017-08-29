#ifndef ONEFLOW_CORE_COMMON_PP_SEQ_FOR_EACH_H_
#define ONEFLOW_CORE_COMMON_PP_SEQ_FOR_EACH_H_
#include "oneflow/core/common/pp_base.h"
#include "oneflow/core/common/pp_seq_size.h"

#define OF_PP_SEQ_FOR_EACH_TUPLE OF_PP_D1_SEQ_FOR_EACH_TUPLE

#define OF_PP_D1_SEQ_FOR_EACH_TUPLE(m, d, seq) \
  OF_PP_D1_SEQ_FOR_EACH(OF_PP_APPLY_TUPLE, m, d, seq)
#define OF_PP_D2_SEQ_FOR_EACH_TUPLE(m, d, seq) \
  OF_PP_D2_SEQ_FOR_EACH(OF_PP_APPLY_TUPLE, m, d, seq)

#define OF_PP_SEQ_FOR_EACH_ATOMIC OF_PP_D1_SEQ_FOR_EACH_ATOMIC

#define OF_PP_D1_SEQ_FOR_EACH_ATOMIC(m, d, seq) \
  OF_PP_D1_SEQ_FOR_EACH(OF_PP_APPLY_ATOMIC, m, d, seq)
#define OF_PP_D2_SEQ_FOR_EACH_ATOMIC(m, d, seq) \
  OF_PP_D2_SEQ_FOR_EACH(OF_PP_APPLY_ATOMIC, m, d, seq)

#define OF_PP_APPLY_TUPLE(m, d, t) m t
#define OF_PP_APPLY_ATOMIC(m, d, x) m(x)
#define OF_PP_APPLY_ATOMIC_WITH_DATA(m, d, x) m(d, x)

#define OF_PP_D1_SEQ_FOR_EACH(apply, m, d, seq) \
  OF_PP_CAT(OF_PP_D1_SEQ_FOR_EACH_, OF_PP_SEQ_SIZE(seq))(apply, m, d, seq)

#define OF_PP_D2_SEQ_FOR_EACH(apply, m, d, seq) \
  OF_PP_CAT(OF_PP_D2_SEQ_FOR_EACH_, OF_PP_SEQ_SIZE(seq))(apply, m, d, seq)

#define OF_PP_D1_SEQ_FOR_EACH_0(apply, m, d, seq)
#define OF_PP_D2_SEQ_FOR_EACH_0(apply, m, d, seq)

#define OF_PP_D1_SEQ_FOR_EACH_1(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                \
      OF_PP_D1_SEQ_FOR_EACH_0(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_1(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                \
      OF_PP_D2_SEQ_FOR_EACH_0(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_2(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                \
      OF_PP_D1_SEQ_FOR_EACH_1(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_2(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                \
      OF_PP_D2_SEQ_FOR_EACH_1(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_3(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                \
      OF_PP_D1_SEQ_FOR_EACH_2(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_3(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                \
      OF_PP_D2_SEQ_FOR_EACH_2(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_4(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                \
      OF_PP_D1_SEQ_FOR_EACH_3(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_4(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                \
      OF_PP_D2_SEQ_FOR_EACH_3(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_5(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                \
      OF_PP_D1_SEQ_FOR_EACH_4(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_5(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                \
      OF_PP_D2_SEQ_FOR_EACH_4(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_6(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                \
      OF_PP_D1_SEQ_FOR_EACH_5(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_6(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                \
      OF_PP_D2_SEQ_FOR_EACH_5(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_7(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                \
      OF_PP_D1_SEQ_FOR_EACH_6(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_7(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                \
      OF_PP_D2_SEQ_FOR_EACH_6(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_8(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                \
      OF_PP_D1_SEQ_FOR_EACH_7(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_8(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                \
      OF_PP_D2_SEQ_FOR_EACH_7(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_9(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                \
      OF_PP_D1_SEQ_FOR_EACH_8(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_9(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                \
      OF_PP_D2_SEQ_FOR_EACH_8(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_10(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_9(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_10(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_9(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_11(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_10(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_11(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_10(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_12(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_11(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_12(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_11(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_13(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_12(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_13(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_12(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_14(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_13(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_14(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_13(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_15(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_14(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_15(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_14(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_16(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_15(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_16(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_15(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_17(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_16(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_17(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_16(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_18(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_17(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_18(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_17(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_19(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_18(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_19(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_18(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_20(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_19(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_20(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_19(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_21(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_20(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_21(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_20(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_22(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_21(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_22(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_21(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_23(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_22(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_23(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_22(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_24(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_23(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_24(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_23(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_25(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_24(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_25(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_24(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_26(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_25(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_26(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_25(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_27(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_26(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_27(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_26(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_28(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_27(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_28(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_27(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_29(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_28(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_29(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_28(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_30(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_29(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_30(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_29(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_31(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_30(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_31(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_30(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_32(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_31(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_32(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_31(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_33(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_32(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_33(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_32(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_34(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_33(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_34(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_33(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_35(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_34(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_35(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_34(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_36(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_35(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_36(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_35(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_37(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_36(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_37(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_36(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_38(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_37(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_38(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_37(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_39(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_38(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_39(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_38(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_40(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_39(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_40(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_39(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_41(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_40(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_41(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_40(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_42(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_41(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_42(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_41(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_43(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_42(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_43(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_42(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_44(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_43(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_44(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_43(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_45(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_44(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_45(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_44(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_46(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_45(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_46(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_45(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_47(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_46(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_47(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_46(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_48(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_47(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_48(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_47(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_49(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_48(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_49(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_48(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_50(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_49(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_50(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_49(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_51(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_50(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_51(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_50(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_52(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_51(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_52(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_51(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_53(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_52(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_53(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_52(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_54(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_53(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_54(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_53(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_55(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_54(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_55(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_54(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_56(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_55(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_56(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_55(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_57(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_56(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_57(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_56(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_58(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_57(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_58(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_57(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_59(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_58(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_59(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_58(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_60(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_59(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_60(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_59(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_61(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_60(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_61(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_60(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_62(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_61(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_62(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_61(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_63(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_62(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_63(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_62(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_64(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_63(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_64(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_63(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_65(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_64(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_65(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_64(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_66(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_65(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_66(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_65(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_67(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_66(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_67(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_66(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_68(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_67(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_68(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_67(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_69(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_68(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_69(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_68(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_70(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_69(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_70(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_69(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_71(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_70(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_71(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_70(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_72(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_71(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_72(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_71(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_73(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_72(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_73(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_72(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_74(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_73(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_74(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_73(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_75(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_74(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_75(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_74(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_76(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_75(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_76(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_75(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_77(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_76(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_77(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_76(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_78(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_77(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_78(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_77(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_79(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_78(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_79(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_78(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_80(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_79(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_80(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_79(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_81(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_80(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_81(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_80(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_82(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_81(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_82(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_81(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_83(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_82(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_83(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_82(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_84(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_83(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_84(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_83(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_85(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_84(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_85(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_84(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_86(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_85(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_86(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_85(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_87(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_86(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_87(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_86(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_88(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_87(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_88(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_87(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_89(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_88(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_89(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_88(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_90(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_89(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_90(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_89(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_91(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_90(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_91(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_90(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_92(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_91(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_92(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_91(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_93(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_92(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_93(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_92(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_94(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_93(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_94(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_93(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_95(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_94(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_95(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_94(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_96(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_95(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_96(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_95(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_97(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_96(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_97(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_96(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_98(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_97(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_98(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_97(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_99(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D1_SEQ_FOR_EACH_98(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_99(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                 \
      OF_PP_D2_SEQ_FOR_EACH_98(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_100(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_99(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_100(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_99(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_101(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_100(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_101(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_100(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_102(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_101(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_102(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_101(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_103(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_102(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_103(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_102(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_104(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_103(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_104(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_103(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_105(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_104(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_105(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_104(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_106(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_105(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_106(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_105(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_107(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_106(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_107(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_106(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_108(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_107(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_108(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_107(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_109(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_108(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_109(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_108(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_110(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_109(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_110(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_109(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_111(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_110(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_111(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_110(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_112(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_111(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_112(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_111(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_113(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_112(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_113(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_112(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_114(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_113(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_114(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_113(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_115(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_114(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_115(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_114(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_116(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_115(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_116(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_115(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_117(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_116(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_117(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_116(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_118(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_117(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_118(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_117(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_119(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_118(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_119(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_118(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_120(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_119(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_120(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_119(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_121(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_120(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_121(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_120(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_122(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_121(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_122(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_121(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_123(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_122(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_123(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_122(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_124(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_123(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_124(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_123(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_125(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_124(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_125(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_124(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_126(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_125(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_126(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_125(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_127(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_126(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_127(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_126(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_128(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_127(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_128(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_127(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_129(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_128(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_129(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_128(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_130(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_129(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_130(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_129(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_131(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_130(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_131(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_130(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_132(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_131(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_132(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_131(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_133(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_132(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_133(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_132(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_134(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_133(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_134(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_133(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_135(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_134(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_135(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_134(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_136(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_135(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_136(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_135(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_137(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_136(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_137(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_136(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_138(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_137(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_138(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_137(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_139(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_138(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_139(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_138(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_140(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_139(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_140(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_139(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_141(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_140(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_141(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_140(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_142(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_141(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_142(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_141(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_143(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_142(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_143(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_142(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_144(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_143(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_144(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_143(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_145(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_144(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_145(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_144(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_146(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_145(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_146(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_145(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_147(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_146(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_147(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_146(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_148(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_147(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_148(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_147(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_149(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_148(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_149(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_148(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_150(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_149(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_150(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_149(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_151(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_150(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_151(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_150(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_152(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_151(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_152(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_151(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_153(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_152(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_153(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_152(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_154(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_153(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_154(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_153(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_155(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_154(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_155(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_154(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_156(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_155(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_156(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_155(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_157(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_156(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_157(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_156(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_158(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_157(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_158(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_157(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_159(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_158(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_159(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_158(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_160(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_159(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_160(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_159(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_161(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_160(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_161(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_160(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_162(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_161(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_162(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_161(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_163(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_162(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_163(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_162(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_164(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_163(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_164(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_163(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_165(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_164(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_165(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_164(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_166(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_165(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_166(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_165(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_167(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_166(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_167(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_166(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_168(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_167(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_168(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_167(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_169(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_168(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_169(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_168(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_170(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_169(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_170(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_169(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_171(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_170(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_171(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_170(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_172(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_171(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_172(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_171(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_173(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_172(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_173(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_172(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_174(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_173(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_174(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_173(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_175(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_174(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_175(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_174(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_176(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_175(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_176(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_175(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_177(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_176(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_177(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_176(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_178(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_177(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_178(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_177(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_179(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_178(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_179(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_178(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_180(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_179(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_180(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_179(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_181(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_180(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_181(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_180(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_182(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_181(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_182(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_181(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_183(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_182(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_183(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_182(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_184(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_183(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_184(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_183(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_185(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_184(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_185(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_184(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_186(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_185(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_186(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_185(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_187(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_186(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_187(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_186(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_188(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_187(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_188(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_187(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_189(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_188(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_189(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_188(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_190(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_189(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_190(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_189(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_191(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_190(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_191(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_190(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_192(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_191(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_192(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_191(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_193(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_192(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_193(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_192(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_194(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_193(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_194(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_193(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_195(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_194(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_195(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_194(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_196(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_195(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_196(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_195(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_197(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_196(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_197(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_196(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_198(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_197(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_198(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_197(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_199(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_198(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_199(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_198(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_200(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_199(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_200(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_199(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_201(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_200(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_201(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_200(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_202(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_201(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_202(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_201(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_203(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_202(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_203(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_202(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_204(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_203(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_204(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_203(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_205(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_204(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_205(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_204(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_206(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_205(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_206(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_205(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_207(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_206(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_207(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_206(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_208(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_207(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_208(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_207(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_209(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_208(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_209(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_208(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_210(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_209(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_210(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_209(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_211(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_210(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_211(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_210(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_212(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_211(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_212(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_211(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_213(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_212(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_213(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_212(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_214(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_213(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_214(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_213(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_215(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_214(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_215(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_214(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_216(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_215(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_216(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_215(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_217(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_216(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_217(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_216(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_218(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_217(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_218(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_217(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_219(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_218(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_219(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_218(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_220(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_219(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_220(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_219(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_221(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_220(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_221(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_220(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_222(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_221(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_222(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_221(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_223(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_222(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_223(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_222(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_224(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_223(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_224(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_223(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_225(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_224(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_225(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_224(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_226(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_225(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_226(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_225(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_227(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_226(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_227(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_226(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_228(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_227(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_228(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_227(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_229(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_228(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_229(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_228(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_230(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_229(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_230(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_229(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_231(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_230(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_231(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_230(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_232(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_231(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_232(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_231(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_233(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_232(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_233(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_232(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_234(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_233(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_234(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_233(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_235(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_234(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_235(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_234(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_236(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_235(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_236(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_235(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_237(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_236(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_237(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_236(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_238(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_237(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_238(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_237(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_239(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_238(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_239(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_238(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_240(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_239(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_240(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_239(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_241(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_240(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_241(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_240(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_242(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_241(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_242(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_241(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_243(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_242(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_243(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_242(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_244(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_243(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_244(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_243(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_245(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_244(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_245(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_244(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_246(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_245(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_246(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_245(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_247(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_246(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_247(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_246(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_248(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_247(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_248(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_247(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_249(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_248(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_249(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_248(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_250(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_249(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_250(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_249(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_251(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_250(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_251(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_250(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_252(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_251(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_252(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_251(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_253(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_252(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_253(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_252(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_254(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_253(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_254(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_253(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_255(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_254(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_255(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_254(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_256(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_255(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_256(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_255(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_257(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_256(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_257(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_256(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_258(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_257(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_258(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_257(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_259(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_258(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_259(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_258(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_260(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_259(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_260(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_259(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_261(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_260(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_261(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_260(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_262(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_261(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_262(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_261(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_263(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_262(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_263(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_262(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_264(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_263(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_264(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_263(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_265(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_264(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_265(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_264(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_266(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_265(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_266(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_265(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_267(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_266(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_267(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_266(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_268(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_267(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_268(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_267(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_269(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_268(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_269(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_268(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_270(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_269(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_270(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_269(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_271(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_270(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_271(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_270(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_272(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_271(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_272(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_271(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_273(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_272(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_273(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_272(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_274(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_273(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_274(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_273(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_275(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_274(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_275(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_274(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_276(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_275(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_276(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_275(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_277(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_276(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_277(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_276(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_278(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_277(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_278(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_277(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_279(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_278(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_279(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_278(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_280(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_279(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_280(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_279(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_281(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_280(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_281(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_280(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_282(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_281(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_282(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_281(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_283(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_282(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_283(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_282(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_284(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_283(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_284(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_283(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_285(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_284(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_285(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_284(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_286(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_285(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_286(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_285(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_287(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_286(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_287(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_286(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_288(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_287(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_288(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_287(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_289(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_288(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_289(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_288(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_290(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_289(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_290(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_289(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_291(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_290(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_291(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_290(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_292(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_291(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_292(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_291(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_293(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_292(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_293(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_292(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_294(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_293(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_294(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_293(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_295(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_294(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_295(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_294(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_296(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_295(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_296(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_295(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_297(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_296(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_297(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_296(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_298(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_297(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_298(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_297(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_299(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_298(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_299(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_298(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_300(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_299(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_300(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_299(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_301(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_300(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_301(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_300(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_302(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_301(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_302(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_301(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_303(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_302(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_303(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_302(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_304(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_303(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_304(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_303(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_305(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_304(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_305(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_304(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_306(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_305(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_306(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_305(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_307(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_306(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_307(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_306(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_308(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_307(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_308(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_307(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_309(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_308(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_309(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_308(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_310(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_309(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_310(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_309(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_311(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_310(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_311(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_310(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_312(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_311(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_312(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_311(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_313(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_312(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_313(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_312(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_314(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_313(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_314(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_313(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_315(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_314(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_315(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_314(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_316(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_315(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_316(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_315(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_317(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_316(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_317(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_316(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_318(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_317(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_318(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_317(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_319(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_318(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_319(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_318(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_320(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_319(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_320(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_319(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_321(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_320(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_321(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_320(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_322(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_321(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_322(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_321(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_323(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_322(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_323(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_322(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_324(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_323(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_324(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_323(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_325(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_324(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_325(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_324(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_326(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_325(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_326(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_325(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_327(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_326(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_327(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_326(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_328(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_327(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_328(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_327(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_329(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_328(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_329(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_328(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_330(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_329(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_330(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_329(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_331(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_330(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_331(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_330(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_332(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_331(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_332(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_331(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_333(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_332(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_333(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_332(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_334(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_333(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_334(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_333(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_335(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_334(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_335(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_334(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_336(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_335(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_336(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_335(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_337(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_336(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_337(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_336(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_338(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_337(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_338(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_337(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_339(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_338(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_339(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_338(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_340(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_339(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_340(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_339(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_341(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_340(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_341(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_340(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_342(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_341(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_342(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_341(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_343(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_342(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_343(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_342(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_344(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_343(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_344(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_343(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_345(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_344(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_345(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_344(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_346(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_345(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_346(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_345(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_347(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_346(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_347(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_346(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_348(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_347(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_348(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_347(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_349(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_348(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_349(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_348(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_350(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_349(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_350(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_349(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_351(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_350(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_351(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_350(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_352(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_351(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_352(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_351(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_353(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_352(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_353(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_352(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_354(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_353(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_354(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_353(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_355(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_354(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_355(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_354(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_356(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_355(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_356(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_355(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_357(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_356(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_357(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_356(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_358(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_357(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_358(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_357(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_359(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_358(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_359(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_358(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_360(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_359(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_360(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_359(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_361(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_360(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_361(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_360(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_362(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_361(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_362(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_361(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_363(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_362(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_363(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_362(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_364(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_363(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_364(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_363(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_365(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_364(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_365(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_364(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_366(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_365(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_366(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_365(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_367(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_366(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_367(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_366(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_368(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_367(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_368(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_367(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_369(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_368(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_369(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_368(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_370(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_369(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_370(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_369(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_371(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_370(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_371(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_370(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_372(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_371(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_372(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_371(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_373(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_372(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_373(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_372(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_374(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_373(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_374(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_373(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_375(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_374(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_375(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_374(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_376(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_375(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_376(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_375(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_377(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_376(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_377(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_376(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_378(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_377(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_378(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_377(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_379(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_378(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_379(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_378(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_380(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_379(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_380(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_379(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_381(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_380(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_381(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_380(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_382(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_381(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_382(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_381(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_383(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_382(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_383(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_382(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_384(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_383(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_384(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_383(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_385(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_384(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_385(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_384(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_386(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_385(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_386(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_385(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_387(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_386(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_387(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_386(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_388(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_387(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_388(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_387(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_389(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_388(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_389(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_388(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_390(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_389(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_390(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_389(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_391(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_390(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_391(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_390(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_392(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_391(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_392(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_391(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_393(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_392(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_393(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_392(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_394(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_393(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_394(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_393(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_395(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_394(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_395(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_394(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_396(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_395(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_396(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_395(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_397(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_396(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_397(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_396(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_398(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_397(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_398(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_397(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_399(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_398(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_399(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_398(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_400(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_399(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_400(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_399(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_401(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_400(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_401(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_400(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_402(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_401(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_402(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_401(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_403(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_402(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_403(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_402(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_404(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_403(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_404(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_403(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_405(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_404(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_405(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_404(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_406(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_405(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_406(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_405(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_407(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_406(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_407(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_406(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_408(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_407(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_408(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_407(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_409(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_408(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_409(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_408(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_410(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_409(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_410(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_409(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_411(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_410(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_411(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_410(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_412(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_411(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_412(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_411(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_413(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_412(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_413(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_412(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_414(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_413(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_414(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_413(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_415(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_414(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_415(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_414(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_416(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_415(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_416(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_415(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_417(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_416(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_417(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_416(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_418(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_417(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_418(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_417(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_419(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_418(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_419(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_418(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_420(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_419(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_420(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_419(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_421(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_420(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_421(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_420(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_422(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_421(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_422(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_421(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_423(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_422(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_423(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_422(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_424(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_423(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_424(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_423(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_425(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_424(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_425(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_424(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_426(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_425(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_426(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_425(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_427(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_426(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_427(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_426(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_428(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_427(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_428(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_427(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_429(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_428(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_429(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_428(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_430(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_429(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_430(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_429(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_431(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_430(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_431(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_430(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_432(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_431(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_432(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_431(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_433(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_432(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_433(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_432(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_434(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_433(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_434(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_433(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_435(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_434(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_435(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_434(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_436(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_435(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_436(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_435(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_437(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_436(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_437(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_436(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_438(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_437(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_438(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_437(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_439(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_438(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_439(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_438(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_440(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_439(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_440(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_439(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_441(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_440(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_441(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_440(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_442(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_441(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_442(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_441(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_443(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_442(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_443(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_442(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_444(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_443(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_444(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_443(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_445(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_444(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_445(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_444(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_446(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_445(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_446(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_445(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_447(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_446(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_447(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_446(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_448(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_447(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_448(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_447(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_449(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_448(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_449(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_448(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_450(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_449(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_450(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_449(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_451(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_450(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_451(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_450(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_452(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_451(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_452(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_451(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_453(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_452(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_453(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_452(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_454(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_453(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_454(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_453(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_455(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_454(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_455(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_454(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_456(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_455(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_456(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_455(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_457(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_456(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_457(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_456(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_458(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_457(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_458(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_457(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_459(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_458(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_459(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_458(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_460(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_459(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_460(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_459(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_461(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_460(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_461(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_460(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_462(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_461(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_462(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_461(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_463(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_462(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_463(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_462(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_464(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_463(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_464(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_463(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_465(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_464(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_465(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_464(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_466(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_465(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_466(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_465(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_467(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_466(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_467(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_466(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_468(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_467(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_468(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_467(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_469(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_468(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_469(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_468(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_470(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_469(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_470(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_469(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_471(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_470(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_471(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_470(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_472(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_471(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_472(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_471(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_473(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_472(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_473(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_472(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_474(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_473(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_474(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_473(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_475(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_474(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_475(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_474(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_476(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_475(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_476(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_475(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_477(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_476(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_477(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_476(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_478(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_477(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_478(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_477(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_479(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_478(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_479(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_478(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_480(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_479(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_480(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_479(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_481(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_480(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_481(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_480(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_482(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_481(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_482(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_481(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_483(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_482(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_483(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_482(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_484(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_483(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_484(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_483(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_485(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_484(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_485(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_484(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_486(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_485(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_486(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_485(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_487(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_486(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_487(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_486(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_488(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_487(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_488(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_487(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_489(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_488(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_489(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_488(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_490(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_489(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_490(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_489(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_491(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_490(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_491(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_490(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_492(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_491(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_492(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_491(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_493(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_492(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_493(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_492(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_494(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_493(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_494(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_493(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_495(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_494(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_495(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_494(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_496(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_495(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_496(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_495(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_497(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_496(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_497(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_496(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_498(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_497(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_498(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_497(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_499(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_498(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_499(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_498(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_500(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_499(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_500(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_499(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_501(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_500(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_501(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_500(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_502(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_501(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_502(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_501(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_503(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_502(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_503(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_502(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_504(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_503(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_504(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_503(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_505(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_504(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_505(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_504(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_506(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_505(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_506(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_505(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_507(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_506(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_507(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_506(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_508(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_507(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_508(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_507(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_509(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_508(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_509(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_508(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_510(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_509(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_510(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_509(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_511(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_510(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_511(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_510(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D1_SEQ_FOR_EACH_512(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D1_SEQ_FOR_EACH_511(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_D2_SEQ_FOR_EACH_512(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))                  \
      OF_PP_D2_SEQ_FOR_EACH_511(apply, m, d, OF_PP_SEQ_TAIL(seq))

#endif  // ONEFLOW_CORE_COMMON_PP_SEQ_FOR_EACH_H_
