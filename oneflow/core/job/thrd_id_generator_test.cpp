#include "oneflow/core/job/thrd_id_generator.h"

namespace oneflow {

TEST(ThrdIdGenerator, ordered) {
  std::vector<std::pair<int64_t, TaskType>> machine_task_type_vec;
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kMdSave));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kMdSave));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kMdSave));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kMdSave));

  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kPrint));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kPrint));

  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kRecordLoad));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kRecordLoad));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kRecordLoad));

  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kLossPrint));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kLossPrint));

  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kMdSave));
  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kMdSave));
  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kMdSave));

  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kPrint));
  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kPrint));
  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kPrint));

  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kLossPrint));
  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kLossPrint));

  int64_t base_thrd_id = 16;
  auto temp = machine_task_type_vec;
  ThrdIdGenerator generator(temp, base_thrd_id + 1);

  std::vector<std::vector<int64_t>> thrd_ids(2);
  for (auto pair : machine_task_type_vec) {
    int64_t thrd_id = generator.GenerateThrdId(pair.first, pair.second);
    thrd_ids[pair.first].push_back(thrd_id);
  }

  std::vector<int64_t> machine0_thrd_ids{/*MdSave*/ 17,     18, 19, 20,
                                         /*Print*/ 21,      22,
                                         /*RecordLoad*/ 23, 24, 25,
                                         /*LossPrint*/ 26,  27};
  std::vector<int64_t> machine1_thrd_ids{/*MdSave*/ 17,    18, 19,
                                         /*Print*/ 20,     21, 22,
                                         /*LossPrint*/ 23, 24};

  CHECK_EQ(thrd_ids[0].size(), machine0_thrd_ids.size());
  CHECK(std::equal(thrd_ids[0].begin(), thrd_ids[0].end(), machine0_thrd_ids.begin()));

  CHECK_EQ(thrd_ids[1].size(), machine1_thrd_ids.size());
  CHECK(std::equal(thrd_ids[1].begin(), thrd_ids[1].end(), machine1_thrd_ids.begin()));
}

TEST(ThrdIdGenerator, disordered) {
  std::vector<std::pair<int64_t, TaskType>> machine_task_type_vec;
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kMdSave));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kMdSave));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kPrint));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kPrint));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kMdSave));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kRecordLoad));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kMdSave));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kRecordLoad));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kLossPrint));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kRecordLoad));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kLossPrint));

  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kMdSave));
  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kPrint));
  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kMdSave));
  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kLossPrint));
  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kPrint));
  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kLossPrint));
  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kPrint));
  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kMdSave));

  int64_t base_thrd_id = 16;
  auto temp = machine_task_type_vec;
  ThrdIdGenerator generator(temp, base_thrd_id + 1);

  std::vector<std::vector<int64_t>> thrd_ids(2);
  for (auto pair : machine_task_type_vec) {
    int64_t thrd_id = generator.GenerateThrdId(pair.first, pair.second);
    thrd_ids[pair.first].push_back(thrd_id);
  }

  std::vector<int64_t> machine0_thrd_ids{17, 18, 21, 22, 19, 23, 20, 24, 26, 25, 27};
  std::vector<int64_t> machine1_thrd_ids{17, 20, 18, 23, 21, 24, 22, 19};

  CHECK_EQ(thrd_ids[0].size(), machine0_thrd_ids.size());
  CHECK(std::equal(thrd_ids[0].begin(), thrd_ids[0].end(), machine0_thrd_ids.begin()));

  CHECK_EQ(thrd_ids[1].size(), machine1_thrd_ids.size());
  CHECK(std::equal(thrd_ids[1].begin(), thrd_ids[1].end(), machine1_thrd_ids.begin()));
}

TEST(ThrdIdGenerator, ordered_mdsave_num_more_than_config) {
  std::vector<std::pair<int64_t, TaskType>> machine_task_type_vec;
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kPrint));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kPrint));

  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kMdSave));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kMdSave));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kMdSave));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kMdSave));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kMdSave));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kMdSave));

  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kRecordLoad));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kRecordLoad));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kRecordLoad));

  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kLossPrint));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kLossPrint));

  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kMdSave));
  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kMdSave));
  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kMdSave));
  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kMdSave));
  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kMdSave));

  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kPrint));
  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kPrint));
  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kPrint));

  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kLossPrint));
  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kLossPrint));

  int64_t base_thrd_id = 16;
  auto temp = machine_task_type_vec;
  // it should be set the mdsave config num for 4 to test
  ThrdIdGenerator generator(temp, base_thrd_id + 1);

  std::vector<std::vector<int64_t>> thrd_ids(2);
  for (auto pair : machine_task_type_vec) {
    int64_t thrd_id = generator.GenerateThrdId(pair.first, pair.second);
    thrd_ids[pair.first].push_back(thrd_id);
  }

  std::vector<int64_t> machine0_thrd_ids{/*Print*/ 17,      18,
                                         /*MdSave*/ 19,     20, 21, 22, 19, 20,
                                         /*RecordLoad*/ 23, 24, 25,
                                         /*LossPrint*/ 26,  27};
  std::vector<int64_t> machine1_thrd_ids{/*MdSave*/ 17,    18, 19, 20, 17,
                                         /*Print*/ 21,     22, 23,
                                         /*LossPrint*/ 24, 25};

  CHECK_EQ(thrd_ids[0].size(), machine0_thrd_ids.size());
  CHECK(std::equal(thrd_ids[0].begin(), thrd_ids[0].end(), machine0_thrd_ids.begin()));

  CHECK_EQ(thrd_ids[1].size(), machine1_thrd_ids.size());
  CHECK(std::equal(thrd_ids[1].begin(), thrd_ids[1].end(), machine1_thrd_ids.begin()));
}

TEST(ThrdIdGenerator, disordered_mdsave_num_more_than_config) {
  std::vector<std::pair<int64_t, TaskType>> machine_task_type_vec;
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kPrint));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kMdSave));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kPrint));

  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kMdSave));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kMdSave));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kMdSave));

  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kRecordLoad));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kMdSave));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kMdSave));

  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kRecordLoad));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kLossPrint));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kMdSave));

  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kRecordLoad));
  machine_task_type_vec.emplace_back(std::make_pair(0, TaskType::kLossPrint));

  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kMdSave));
  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kPrint));
  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kMdSave));

  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kLossPrint));
  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kPrint));
  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kMdSave));

  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kLossPrint));
  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kMdSave));
  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kPrint));

  machine_task_type_vec.emplace_back(std::make_pair(1, TaskType::kMdSave));

  int64_t base_thrd_id = 16;
  auto temp = machine_task_type_vec;
  // it should be set the mdsave config num for 4 to test
  ThrdIdGenerator generator(temp, base_thrd_id + 1);

  std::vector<std::vector<int64_t>> thrd_ids(2);
  for (auto pair : machine_task_type_vec) {
    int64_t thrd_id = generator.GenerateThrdId(pair.first, pair.second);
    thrd_ids[pair.first].push_back(thrd_id);
  }

  std::vector<int64_t> machine0_thrd_ids{17, 19, 18, 20, 21, 22, 23, 19, 20, 24, 26, 21, 25, 27};
  std::vector<int64_t> machine1_thrd_ids{17, 21, 18, 24, 22, 19, 25, 20, 23, 17};
  CHECK_EQ(thrd_ids[0].size(), machine0_thrd_ids.size());
  CHECK(std::equal(thrd_ids[0].begin(), thrd_ids[0].end(), machine0_thrd_ids.begin()));

  CHECK_EQ(thrd_ids[1].size(), machine1_thrd_ids.size());
  CHECK(std::equal(thrd_ids[1].begin(), thrd_ids[1].end(), machine1_thrd_ids.begin()));
}

}  // namespace oneflow
