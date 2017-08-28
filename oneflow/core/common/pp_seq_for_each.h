#ifndef ONEFLOW_CORE_COMMON_PP_SEQ_FOR_EACH_H_
#define ONEFLOW_CORE_COMMON_PP_SEQ_FOR_EACH_H_
#include "oneflow/core/common/pp_base.h"
#include "oneflow/core/common/pp_seq_size.h"

#define OF_PP_SEQ_FOR_EACH_TUPLE(m, d, seq) \
  OF_PP_SEQ_FOR_EACH(OF_PP_APPLY_TUPLE, m, d, seq)
#define OF_PP_APPLY_TUPLE(m, d, t) m t

#define OF_PP_SEQ_FOR_EACH_ATOMIC(m, d, seq) \
  OF_PP_SEQ_FOR_EACH(OF_PP_APPLY_ATOMIC, m, d, seq)
#define OF_PP_APPLY_ATOMIC(m, d, x) m(x)

#define OF_PP_SEQ_FOR_EACH(apply, m, d, seq) \
  OF_PP_CAT(OF_PP_SEQ_FOR_EACH_, OF_PP_SEQ_SIZE(seq))(apply, m, d, seq)
#define OF_PP_SEQ_FOR_EACH_0(apply, m, d, seq)
#define OF_PP_SEQ_FOR_EACH_1(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))             \
      OF_PP_SEQ_FOR_EACH_0(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_2(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))             \
      OF_PP_SEQ_FOR_EACH_1(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_3(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))             \
      OF_PP_SEQ_FOR_EACH_2(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_4(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))             \
      OF_PP_SEQ_FOR_EACH_3(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_5(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))             \
      OF_PP_SEQ_FOR_EACH_4(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_6(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))             \
      OF_PP_SEQ_FOR_EACH_5(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_7(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))             \
      OF_PP_SEQ_FOR_EACH_6(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_8(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))             \
      OF_PP_SEQ_FOR_EACH_7(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_9(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))             \
      OF_PP_SEQ_FOR_EACH_8(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_10(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_9(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_11(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_10(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_12(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_11(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_13(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_12(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_14(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_13(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_15(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_14(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_16(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_15(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_17(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_16(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_18(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_17(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_19(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_18(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_20(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_19(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_21(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_20(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_22(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_21(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_23(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_22(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_24(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_23(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_25(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_24(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_26(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_25(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_27(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_26(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_28(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_27(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_29(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_28(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_30(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_29(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_31(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_30(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_32(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_31(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_33(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_32(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_34(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_33(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_35(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_34(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_36(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_35(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_37(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_36(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_38(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_37(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_39(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_38(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_40(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_39(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_41(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_40(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_42(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_41(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_43(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_42(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_44(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_43(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_45(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_44(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_46(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_45(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_47(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_46(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_48(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_47(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_49(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_48(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_50(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_49(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_51(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_50(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_52(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_51(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_53(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_52(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_54(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_53(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_55(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_54(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_56(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_55(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_57(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_56(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_58(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_57(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_59(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_58(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_60(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_59(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_61(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_60(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_62(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_61(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_63(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_62(apply, m, d, OF_PP_SEQ_TAIL(seq))
#define OF_PP_SEQ_FOR_EACH_64(apply, m, d, seq) \
  apply(m, d, OF_PP_SEQ_HEAD(seq))              \
      OF_PP_SEQ_FOR_EACH_63(apply, m, d, OF_PP_SEQ_TAIL(seq))

#endif  // ONEFLOW_CORE_COMMON_PP_SEQ_FOR_EACH_H_
