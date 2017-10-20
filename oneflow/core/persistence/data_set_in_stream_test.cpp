#include "oneflow/core/common/process_state.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/persistence/cyclic_data_set_in_stream.h"
#include "oneflow/core/persistence/normal_data_set_in_stream.h"

namespace oneflow {

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

TEST(DataSetInStream, normal_naive) {
  uint32_t random = NewRandomSeed();
  NormalDataSetInStream in_stream(LocalFS(), SavedFile(random));
  auto record = FlexibleMalloc<Record>(0);
  int ret = in_stream.ReadRecord(&record);
  ASSERT_EQ(ret, 0);
  ASSERT_EQ(std::to_string(random), record->GetKey());
  ASSERT_EQ(random, *reinterpret_cast<const uint32_t*>(record->value_buffer()));

  ret = in_stream.ReadRecord(&record);
  ASSERT_TRUE(ret < 0);
}

TEST(DataSetInStream, cyclic_naive) {
  uint32_t random = NewRandomSeed();
  CyclicDataSetInStream in_stream(LocalFS(), SavedFile(random));
  auto record = FlexibleMalloc<Record>(0);
  int ret;
  for (int i = 0; i < 10; ++i) {
    ret = in_stream.ReadRecord(&record);
    std::cout << "i = " << i << "\tret = " << ret
              << "\t size = " << FlexibleSizeOf(*record) << std::endl;
    ASSERT_EQ(ret, 0);
    ASSERT_EQ(std::to_string(random), record->GetKey());
    ASSERT_EQ(random,
              *reinterpret_cast<const uint32_t*>(record->value_buffer()));
  }
}

}  // namespace oneflow
