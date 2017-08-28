#include "oneflow/core/common/pp_seq.h"
#include <unordered_map>
#include "gtest/gtest.h"

namespace oneflow {

TEST(PP_SEQ, seq_size) {
#define SEQ (1)(2)(3)
  ASSERT_EQ(OF_PP_SEQ_SIZE(SEQ), 3);
#undef SEQ
}

TEST(PP_SEQ, big_seq_size) {
#define SEQ                                                                    \
  (0)(1)(2)(3)(4)(5)(6)(7)(8)(9)(10)(11)(12)(13)(14)(15)(16)(17)(18)(19)(20)(  \
      21)(22)(23)(24)(25)(26)(27)(28)(29)(30)(31)(32)(33)(34)(35)(36)(37)(38)( \
      39)(40)(41)(42)(43)(44)(45)(46)(47)(48)(49)(50)(51)(52)(53)(54)(55)(56)( \
      57)(58)(59)(60)(61)(62)(63)
  ASSERT_EQ(OF_PP_SEQ_SIZE(SEQ), 64);
#undef SEQ
}

TEST(PP_SEQ, for_each) {
#define SEQ (1)(2)(3)(4)
#define MAKE_PAIR(x) {x, x},
  std::unordered_map<int, int> identity = {
      OF_PP_SEQ_FOR_EACH_ATOMIC(MAKE_PAIR, _, SEQ)};
#undef MAKE_PAIR
#undef SEQ
  for (int i = 1; i <= 4; ++i) { ASSERT_EQ(i, identity[i]); }
}

TEST(PP_SEQ, for_each_tuple) {
#define SEQ ((1, 1))((2, 2))((3, 3))((4, 4))
#define MAKE_ENTRY(x, y) {x, y},
  std::unordered_map<int, int> identity = {
      OF_PP_SEQ_FOR_EACH_TUPLE(MAKE_ENTRY, _, SEQ)};
#undef MAKE_ENTRY
#undef SEQ
  for (int i = 1; i <= 4; ++i) { ASSERT_EQ(i, identity[i]); }
}

}  // namespace oneflow
