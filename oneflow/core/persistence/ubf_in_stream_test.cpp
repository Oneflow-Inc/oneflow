#include "oneflow/core/common/process_state.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/persistence/cyclic_ubf_in_stream.h"
#include "oneflow/core/persistence/normal_ubf_in_stream.h"

namespace oneflow {
namespace test {

std::string SavedFile(uint32_t random) {
  std::string current_dir = GetCwd();
  StringReplace(&current_dir, '\\', '/');
  std::string file_name = JoinPath(current_dir, "/tmp_test_cyclic_data_set");
  PersistentOutStream out_stream(LocalFS(), file_name);
  auto item = UbfUtil::CreateLabelItem(std::to_string(random), random);
  out_stream << *item;
  return file_name;
}

TEST(UbfInStream, normal_naive) {
  uint32_t random = NewRandomSeed();
  NormalUbfInStream in_stream(LocalFS(), SavedFile(random));
  std::unique_ptr<UbfItem> ubf_item;
  int ret = in_stream.ReadOneItem(&ubf_item);
  ASSERT_EQ(ret, 0);
  ASSERT_EQ(std::to_string(random), ubf_item->data_id());
  ASSERT_EQ(random, *reinterpret_cast<const uint32_t*>(ubf_item->body()));

  ret = in_stream.ReadOneItem(&ubf_item);
  ASSERT_TRUE(ret < 0);
}

TEST(UbfInStream, cyclic_naive) {
  uint32_t random = NewRandomSeed();
  CyclicUbfInStream in_stream(LocalFS(), SavedFile(random));
  std::unique_ptr<UbfItem> ubf_item;
  int ret;
  for (int i = 0; i < 10; ++i) {
    ret = in_stream.ReadOneItem(&ubf_item);
    ASSERT_EQ(ret, 0);
    ASSERT_EQ(std::to_string(random), ubf_item->data_id());
    ASSERT_EQ(random, *reinterpret_cast<const uint32_t*>(ubf_item->body()));
  }
}

}  // namespace test
}  // namespace oneflow
