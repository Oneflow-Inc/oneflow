#include "oneflow/core/common/static_counter.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace test {

namespace {

DEFINE_STATIC_COUNTER(static_counter);

TEST(StaticCounter, eq0) { static_assert(STATIC_COUNTER(static_counter) == 0, ""); }

INCREASE_STATIC_COUNTER(static_counter);

TEST(StaticCounter, eq1) { static_assert(STATIC_COUNTER(static_counter) == 1, ""); }

TEST(StaticCounter, eq1_again) { static_assert(STATIC_COUNTER(static_counter) == 1, ""); }

INCREASE_STATIC_COUNTER(static_counter);

TEST(StaticCounter, eq2) { static_assert(STATIC_COUNTER(static_counter) == 2, ""); }

}  // namespace

}  // namespace test

}  // namespace oneflow
