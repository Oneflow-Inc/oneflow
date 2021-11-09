/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <gtest/gtest.h>
#include "oneflow/maybe/variant.h"

using namespace oneflow::maybe;

TEST(Variant, Basics) {
  Variant<int, float> a, b(1), c(1.2f), d(InPlaceType<int>, 'a'), e(InPlaceType<float>, 6.66);
  ASSERT_TRUE(a.Is<int>());
  ASSERT_EQ(a.Get<int>(), 0);
  ASSERT_TRUE(b.Is<int>());
  ASSERT_EQ(b.Get<int>(), 1);
  ASSERT_TRUE(c.Is<float>());
  ASSERT_EQ(c.Get<float>(), 1.2f);
  ASSERT_TRUE(d.Is<int>());
  ASSERT_EQ(d.Get<int>(), 'a');
  ASSERT_TRUE(e.Is<float>());
  ASSERT_FLOAT_EQ(e.Get<float>(), 6.66);

  Variant<int, float> f(b), g(c), h(InPlaceIndex<1>, 2.33), i(InPlaceIndex<0>, 2.33);
  ASSERT_TRUE(f.Is<int>());
  ASSERT_EQ(f.Get<int>(), 1);
  ASSERT_TRUE(g.Is<float>());
  ASSERT_EQ(g.Get<float>(), 1.2f);
  ASSERT_TRUE(h.Is<float>());
  ASSERT_FLOAT_EQ(h.Get<float>(), 2.33);
  ASSERT_TRUE(i.Is<int>());
  ASSERT_EQ(i.Get<int>(), 2);

  a = 1;
  ASSERT_TRUE(a.Is<int>());
  ASSERT_EQ(a.Get<int>(), 1);

  a = 1.3f;
  ASSERT_TRUE(a.Is<float>());
  ASSERT_EQ(a.Get<float>(), 1.3f);

  a = b;
  ASSERT_TRUE(a.Is<int>());
  ASSERT_EQ(a.Get<int>(), 1);

  a = c;
  ASSERT_TRUE(a.Is<float>());
  ASSERT_EQ(a.Get<float>(), 1.2f);

  ASSERT_EQ((b.visit<Variant<int, float>>([](auto&& x) { return x + 1; })),
            (Variant<int, float>(2)));
  ASSERT_EQ((c.visit<Variant<int, float>>([](auto&& x) { return x + 1; })),
            (Variant<int, float>(2.2f)));
}
