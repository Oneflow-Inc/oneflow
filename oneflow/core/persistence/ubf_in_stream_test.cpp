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
  auto header = of_make_unique<UbfHeader>("label", 1, std::vector<uint32_t>{1});
  auto item = UbfUtil::CreateLabelItem(std::to_string(random), random);
  out_stream << *header << *item;
  return file_name;
}

TEST(UbfInStream, normal_naive) {
  uint32_t random = NewRandomSeed();
  NormalUbfInStream in_stream(LocalFS(), SavedFile(random));
  auto ubf_item = Flexible<UbfItem>::Malloc(sizeof(uint32_t));
  int ret = in_stream.ReadOneItem(&ubf_item);
  ASSERT_EQ(ret, 0);
  ASSERT_EQ(std::to_string(random), ubf_item->GetDataId());
  ASSERT_EQ(random,
            *reinterpret_cast<const uint32_t*>(ubf_item->value_buffer()));

  ret = in_stream.ReadOneItem(&ubf_item);
  ASSERT_TRUE(ret < 0);
}

TEST(UbfInStream, cyclic_naive) {
  uint32_t random = NewRandomSeed();
  CyclicUbfInStream in_stream(LocalFS(), SavedFile(random));
  auto ubf_item = Flexible<UbfItem>::Malloc(sizeof(uint32_t));
  int ret;
  for (int i = 0; i < 10; ++i) {
    ret = in_stream.ReadOneItem(&ubf_item);
    ASSERT_EQ(ret, 0);
    ASSERT_EQ(std::to_string(random), ubf_item->GetDataId());
    ASSERT_EQ(random,
              *reinterpret_cast<const uint32_t*>(ubf_item->value_buffer()));
  }
}

}  // namespace test
}  // namespace oneflow
