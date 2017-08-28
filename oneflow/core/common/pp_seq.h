#ifndef ONEFLOW_CORE_COMMON_PP_SEQ_H_
#define ONEFLOW_CORE_COMMON_PP_SEQ_H_

#define OF_PP_CAT(a, b) OF_PP_CAT_I(a, b)
#define OF_PP_CAT_I(a, b) a##b

#define OF_PP_SEQ_FOR_EACH_PAIR(macro, seq) \
  OF_PP_FOR_EACH_(OF_PP_RUN_PAIR, macro, seq)
#define OF_PP_RUN_PAIR(macro, pair) macro pair

#define OF_PP_FOR_EACH_(run, macro, seq) \
  OF_PP_CAT(OF_PP_FOR_EACH_, OF_PP_SEQ_SIZE(seq))(run, macro, seq)
#define OF_PP_FOR_EACH_0(run, macro, seq)
#define OF_PP_FOR_EACH_1(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))         \
      OF_PP_FOR_EACH_0(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_2(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))         \
      OF_PP_FOR_EACH_1(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_3(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))         \
      OF_PP_FOR_EACH_2(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_4(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))         \
      OF_PP_FOR_EACH_3(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_5(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))         \
      OF_PP_FOR_EACH_4(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_6(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))         \
      OF_PP_FOR_EACH_5(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_7(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))         \
      OF_PP_FOR_EACH_6(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_8(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))         \
      OF_PP_FOR_EACH_7(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_9(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))         \
      OF_PP_FOR_EACH_8(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_10(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_9(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_11(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_10(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_12(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_11(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_13(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_12(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_14(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_13(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_15(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_14(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_16(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_15(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_17(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_16(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_18(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_17(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_19(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_18(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_20(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_19(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_21(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_20(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_22(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_21(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_23(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_22(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_24(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_23(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_25(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_24(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_26(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_25(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_27(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_26(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_28(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_27(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_29(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_28(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_30(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_29(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_31(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_30(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_32(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_31(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_33(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_32(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_34(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_33(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_35(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_34(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_36(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_35(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_37(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_36(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_38(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_37(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_39(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_38(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_40(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_39(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_41(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_40(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_42(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_41(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_43(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_42(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_44(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_43(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_45(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_44(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_46(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_45(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_47(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_46(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_48(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_47(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_49(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_48(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_50(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_49(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_51(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_50(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_52(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_51(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_53(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_52(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_54(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_53(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_55(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_54(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_56(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_55(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_57(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_56(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_58(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_57(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_59(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_58(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_60(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_59(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_61(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_60(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_62(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_61(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_63(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_62(run, macro, OF_PP_SEQ_TAIL(seq))
#define OF_PP_FOR_EACH_64(run, macro, seq) \
  run(macro, OF_PP_SEQ_HEAD(seq))          \
      OF_PP_FOR_EACH_63(run, macro, OF_PP_SEQ_TAIL(seq))

#define OF_PP_SEQ_HEAD(seq) OF_PP_PAIR_FIRST(OF_PP_SEQ_TO_PAIR(seq))
#define OF_PP_SEQ_TAIL(seq) OF_PP_PAIR_SECOND(OF_PP_SEQ_TO_PAIR(seq))

#define OF_PP_SEQ_TO_PAIR(seq) (OF_PP_SEQ_TO_PAIR_ seq)
#define OF_PP_SEQ_TO_PAIR_(x) x, OF_PP_SEQ_NIL
#define OF_PP_SEQ_NIL

#define OF_PP_PAIR_FIRST(t) OF_PP_FIRST_ARG t
#define OF_PP_PAIR_SECOND(t) OF_PP_SECOND_ARG t

#define OF_PP_FIRST_ARG(x, y) x
#define OF_PP_SECOND_ARG(x, y) y

#define OF_PP_MAKE_PAIR(a, b) (a, b)
#define OF_PP_MAKE_TUPLE(...) (__VA_ARGS__)

#define OF_PP_SEQ_SIZE(seq) OF_PP_SEQ_SIZE_I(seq)
#define OF_PP_SEQ_SIZE_I(seq) OF_PP_CAT(OF_PP_SEQ_SIZE_, OF_PP_SEQ_SIZE_0 seq)
#define OF_PP_SEQ_SIZE_0(_) OF_PP_SEQ_SIZE_1
#define OF_PP_SEQ_SIZE_1(_) OF_PP_SEQ_SIZE_2
#define OF_PP_SEQ_SIZE_2(_) OF_PP_SEQ_SIZE_3
#define OF_PP_SEQ_SIZE_3(_) OF_PP_SEQ_SIZE_4
#define OF_PP_SEQ_SIZE_4(_) OF_PP_SEQ_SIZE_5
#define OF_PP_SEQ_SIZE_5(_) OF_PP_SEQ_SIZE_6
#define OF_PP_SEQ_SIZE_6(_) OF_PP_SEQ_SIZE_7
#define OF_PP_SEQ_SIZE_7(_) OF_PP_SEQ_SIZE_8
#define OF_PP_SEQ_SIZE_8(_) OF_PP_SEQ_SIZE_9
#define OF_PP_SEQ_SIZE_9(_) OF_PP_SEQ_SIZE_10
#define OF_PP_SEQ_SIZE_10(_) OF_PP_SEQ_SIZE_11
#define OF_PP_SEQ_SIZE_11(_) OF_PP_SEQ_SIZE_12
#define OF_PP_SEQ_SIZE_12(_) OF_PP_SEQ_SIZE_13
#define OF_PP_SEQ_SIZE_13(_) OF_PP_SEQ_SIZE_14
#define OF_PP_SEQ_SIZE_14(_) OF_PP_SEQ_SIZE_15
#define OF_PP_SEQ_SIZE_15(_) OF_PP_SEQ_SIZE_16
#define OF_PP_SEQ_SIZE_16(_) OF_PP_SEQ_SIZE_17
#define OF_PP_SEQ_SIZE_17(_) OF_PP_SEQ_SIZE_18
#define OF_PP_SEQ_SIZE_18(_) OF_PP_SEQ_SIZE_19
#define OF_PP_SEQ_SIZE_19(_) OF_PP_SEQ_SIZE_20
#define OF_PP_SEQ_SIZE_20(_) OF_PP_SEQ_SIZE_21
#define OF_PP_SEQ_SIZE_21(_) OF_PP_SEQ_SIZE_22
#define OF_PP_SEQ_SIZE_22(_) OF_PP_SEQ_SIZE_23
#define OF_PP_SEQ_SIZE_23(_) OF_PP_SEQ_SIZE_24
#define OF_PP_SEQ_SIZE_24(_) OF_PP_SEQ_SIZE_25
#define OF_PP_SEQ_SIZE_25(_) OF_PP_SEQ_SIZE_26
#define OF_PP_SEQ_SIZE_26(_) OF_PP_SEQ_SIZE_27
#define OF_PP_SEQ_SIZE_27(_) OF_PP_SEQ_SIZE_28
#define OF_PP_SEQ_SIZE_28(_) OF_PP_SEQ_SIZE_29
#define OF_PP_SEQ_SIZE_29(_) OF_PP_SEQ_SIZE_30
#define OF_PP_SEQ_SIZE_30(_) OF_PP_SEQ_SIZE_31
#define OF_PP_SEQ_SIZE_31(_) OF_PP_SEQ_SIZE_32
#define OF_PP_SEQ_SIZE_32(_) OF_PP_SEQ_SIZE_33
#define OF_PP_SEQ_SIZE_33(_) OF_PP_SEQ_SIZE_34
#define OF_PP_SEQ_SIZE_34(_) OF_PP_SEQ_SIZE_35
#define OF_PP_SEQ_SIZE_35(_) OF_PP_SEQ_SIZE_36
#define OF_PP_SEQ_SIZE_36(_) OF_PP_SEQ_SIZE_37
#define OF_PP_SEQ_SIZE_37(_) OF_PP_SEQ_SIZE_38
#define OF_PP_SEQ_SIZE_38(_) OF_PP_SEQ_SIZE_39
#define OF_PP_SEQ_SIZE_39(_) OF_PP_SEQ_SIZE_40
#define OF_PP_SEQ_SIZE_40(_) OF_PP_SEQ_SIZE_41
#define OF_PP_SEQ_SIZE_41(_) OF_PP_SEQ_SIZE_42
#define OF_PP_SEQ_SIZE_42(_) OF_PP_SEQ_SIZE_43
#define OF_PP_SEQ_SIZE_43(_) OF_PP_SEQ_SIZE_44
#define OF_PP_SEQ_SIZE_44(_) OF_PP_SEQ_SIZE_45
#define OF_PP_SEQ_SIZE_45(_) OF_PP_SEQ_SIZE_46
#define OF_PP_SEQ_SIZE_46(_) OF_PP_SEQ_SIZE_47
#define OF_PP_SEQ_SIZE_47(_) OF_PP_SEQ_SIZE_48
#define OF_PP_SEQ_SIZE_48(_) OF_PP_SEQ_SIZE_49
#define OF_PP_SEQ_SIZE_49(_) OF_PP_SEQ_SIZE_50
#define OF_PP_SEQ_SIZE_50(_) OF_PP_SEQ_SIZE_51
#define OF_PP_SEQ_SIZE_51(_) OF_PP_SEQ_SIZE_52
#define OF_PP_SEQ_SIZE_52(_) OF_PP_SEQ_SIZE_53
#define OF_PP_SEQ_SIZE_53(_) OF_PP_SEQ_SIZE_54
#define OF_PP_SEQ_SIZE_54(_) OF_PP_SEQ_SIZE_55
#define OF_PP_SEQ_SIZE_55(_) OF_PP_SEQ_SIZE_56
#define OF_PP_SEQ_SIZE_56(_) OF_PP_SEQ_SIZE_57
#define OF_PP_SEQ_SIZE_57(_) OF_PP_SEQ_SIZE_58
#define OF_PP_SEQ_SIZE_58(_) OF_PP_SEQ_SIZE_59
#define OF_PP_SEQ_SIZE_59(_) OF_PP_SEQ_SIZE_60
#define OF_PP_SEQ_SIZE_60(_) OF_PP_SEQ_SIZE_61
#define OF_PP_SEQ_SIZE_61(_) OF_PP_SEQ_SIZE_62
#define OF_PP_SEQ_SIZE_62(_) OF_PP_SEQ_SIZE_63
#define OF_PP_SEQ_SIZE_63(_) OF_PP_SEQ_SIZE_64
#define OF_PP_SEQ_SIZE_64(_) OF_PP_SEQ_SIZE_65

#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_0 0
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_1 1
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_2 2
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_3 3
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_4 4
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_5 5
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_6 6
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_7 7
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_8 8
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_9 9
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_10 10
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_11 11
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_12 12
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_13 13
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_14 14
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_15 15
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_16 16
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_17 17
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_18 18
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_19 19
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_20 20
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_21 21
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_22 22
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_23 23
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_24 24
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_25 25
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_26 26
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_27 27
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_28 28
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_29 29
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_30 30
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_31 31
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_32 32
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_33 33
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_34 34
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_35 35
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_36 36
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_37 37
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_38 38
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_39 39
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_40 40
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_41 41
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_42 42
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_43 43
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_44 44
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_45 45
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_46 46
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_47 47
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_48 48
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_49 49
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_50 50
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_51 51
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_52 52
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_53 53
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_54 54
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_55 55
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_56 56
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_57 57
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_58 58
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_59 59
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_60 60
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_61 61
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_62 62
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_63 63
#define OF_PP_SEQ_SIZE_OF_PP_SEQ_SIZE_64 64

#endif  // ONEFLOW_CORE_COMMON_PP_SEQ_H_
