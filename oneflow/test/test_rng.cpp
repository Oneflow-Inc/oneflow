#include <curand.h>
#include <gtest/gtest.h>

#include <cstdlib>

#include "common/rng.h"
namespace caffe {
class RngTest :public::testing::TestWithParam<size_t>{
 protected:
  RngTest():MAXN(100) {
    a = reinterpret_cast<double*>(calloc(MAXN, sizeof(double)));
    b = reinterpret_cast<double*>(calloc(MAXN, sizeof(double)));
  }
  ~RngTest() {
    free(a);
    free(b);
  }
  const int MAXN;
  double* a;
  double* b;
};
size_t test_seed(size_t seed) {
  RNG::set_seed(seed);
  return RNG::get_seed();
}
bool test_seed_generate(double* a, double*b, int n) {
  int noteq = 0;
  for (int i = 0; i < n; ++i) {
    if (a[i] != b[i]) noteq++;
  }
  if (noteq > n - noteq) return true;
  return false;
}
INSTANTIATE_TEST_CASE_P(RngTest_SeedTest_Test, RngTest,
  ::testing::Range(0ULL, 1000ULL, 100));
TEST_P(RngTest, SeedTest) {
  int seed = GetParam();
  ASSERT_EQ(seed, test_seed(seed));
  ASSERT_EQ(curandGenerateUniformDouble(RNG::generator(), b, MAXN),
    CURAND_STATUS_SUCCESS);
  ASSERT_TRUE(test_seed_generate(a, b, MAXN));
  memcpy(a, b, MAXN);
}
TEST(RngTest, TypeTest) {
  ASSERT_EQ(typeid(RNG::generator()), typeid(curandGenerator_t));
}

}  // namespace caffe
