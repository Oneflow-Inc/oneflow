#include "oneflow/core/common/process_state.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/persistence/cyclic_ofb_in_stream.h"
#include "oneflow/core/persistence/normal_ofb_in_stream.h"

namespace oneflow {
namespace test {

std::string SavedFile(uint32_t random) {
  std::string current_dir = GetCwd();
  StringReplace(&current_dir, '\\', '/');
  std::string file_name = JoinPath(current_dir, "/tmp_test_cyclic_data_set");
  PersistentOutStream out_stream(LocalFS(), file_name);
  auto header = DataSetUtil::CreateHeader("label", 1, {1});
  auto item = DataSetUtil::CreateLabelItem(std::to_string(random), random);
  out_stream << *header << *item;
  return file_name;
}

TEST(OfbInStream, normal_naive) {
  uint32_t random = NewRandomSeed();
  NormalOfbInStream in_stream(LocalFS(), SavedFile(random));
  auto ofb_item = FlexibleMalloc<OfbItem>(0);
  int ret = in_stream.ReadOfbItem(&ofb_item);
  ASSERT_EQ(ret, 0);
  ASSERT_EQ(std::to_string(random), ofb_item->GetDataId());
  ASSERT_EQ(random,
            *reinterpret_cast<const uint32_t*>(ofb_item->value_buffer()));

  ret = in_stream.ReadOfbItem(&ofb_item);
  ASSERT_TRUE(ret < 0);
}

TEST(OfbInStream, cyclic_naive) {
  uint32_t random = NewRandomSeed();
  CyclicOfbInStream in_stream(LocalFS(), SavedFile(random));
  auto ofb_item = FlexibleMalloc<OfbItem>(0);
  int ret;
  for (int i = 0; i < 10; ++i) {
    ret = in_stream.ReadOfbItem(&ofb_item);
    ASSERT_EQ(ret, 0);
    ASSERT_EQ(std::to_string(random), ofb_item->GetDataId());
    ASSERT_EQ(random,
              *reinterpret_cast<const uint32_t*>(ofb_item->value_buffer()));
  }
}

}  // namespace test
}  // namespace oneflow
