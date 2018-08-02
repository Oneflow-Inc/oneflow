#include "oneflow/core/job/thrd_id_generator.h"

namespace oneflow {

TEST(ThrdIdGenerator, ordered) {
  HashMap<std::pair<int64_t, int64_t>, int32_t> machine_task_type2thrd_num;
  machine_task_type2thrd_num.emplace(std::make_pair(0, TaskType::kMdSave), 4);
  machine_task_type2thrd_num.emplace(std::make_pair(0, TaskType::kPrint), 2);
  machine_task_type2thrd_num.emplace(std::make_pair(0, TaskType::kRecordLoad), 3);
  machine_task_type2thrd_num.emplace(std::make_pair(0, TaskType::kLossPrint), 2);

  machine_task_type2thrd_num.emplace(std::make_pair(1, TaskType::kMdSave), 3);
  machine_task_type2thrd_num.emplace(std::make_pair(1, TaskType::kPrint), 3);
  machine_task_type2thrd_num.emplace(std::make_pair(1, TaskType::kLossPrint), 2);

  // ordered; 4 mdsave, 2 print, 3 recordload, 2 lossprint
  std::vector<int64_t> vec0{
      // mdsave
      TaskType::kMdSave, TaskType::kMdSave, TaskType::kMdSave, TaskType::kMdSave,
      // print
      TaskType::kPrint, TaskType::kPrint,
      // record load
      TaskType::kRecordLoad, TaskType::kRecordLoad, TaskType::kRecordLoad,
      // lossprint
      TaskType::kLossPrint, TaskType::kLossPrint};

  // 3 mdsave, 3 print, 2 lossprint
  std::vector<int64_t> vec1{// mdsave
                            TaskType::kMdSave, TaskType::kMdSave, TaskType::kMdSave,
                            // print
                            TaskType::kPrint, TaskType::kPrint, TaskType::kPrint,
                            // lossprint
                            TaskType::kLossPrint, TaskType::kLossPrint};

  std::vector<std::vector<int64_t>> machine_task_type{vec0, vec1};
  int64_t base_thrd_id = 16;
  ThrdIdGenerator generator(machine_task_type2thrd_num, base_thrd_id);
  std::vector<std::vector<int64_t>> thrd_ids(machine_task_type.size());
  for (int i = 0; i < machine_task_type.size(); ++i) {
    int64_t machine_id = i;
    std::vector<int64_t> vec = machine_task_type[i];
    for (int64_t task_type : vec) {
      int64_t thrd_id = generator.GenerateThrdId(machine_id, task_type);
      thrd_ids[machine_id].push_back(thrd_id);
    }
  }

  std::vector<int64_t> machine0_thrd_ids{/*MdSave*/ 17,     18, 19, 20, /*Print*/ 21, 22,
                                         /*RecordLoad*/ 23, 24, 25,
                                         /*LossPrint*/ 26,  27};
  std::vector<int64_t> machine1_thrd_ids{/*MdSave*/ 17,    18, 19, /*Print*/ 20, 21, 22,
                                         /*LossPrint*/ 23, 24};

  CHECK_EQ(thrd_ids[0].size(), machine0_thrd_ids.size());
  CHECK(std::equal(thrd_ids[0].begin(), thrd_ids[0].end(), machine0_thrd_ids.begin()));

  CHECK_EQ(thrd_ids[1].size(), machine1_thrd_ids.size());
  CHECK(std::equal(thrd_ids[1].begin(), thrd_ids[1].end(), machine1_thrd_ids.begin()));
}

TEST(ThrdIdGenerator, disordered) {
  HashMap<std::pair<int64_t, int64_t>, int32_t> machine_task_type2thrd_num;
  machine_task_type2thrd_num.emplace(std::make_pair(0, TaskType::kMdSave), 4);
  machine_task_type2thrd_num.emplace(std::make_pair(0, TaskType::kPrint), 2);
  machine_task_type2thrd_num.emplace(std::make_pair(0, TaskType::kRecordLoad), 3);
  machine_task_type2thrd_num.emplace(std::make_pair(0, TaskType::kLossPrint), 2);

  machine_task_type2thrd_num.emplace(std::make_pair(1, TaskType::kMdSave), 3);
  machine_task_type2thrd_num.emplace(std::make_pair(1, TaskType::kPrint), 3);
  machine_task_type2thrd_num.emplace(std::make_pair(1, TaskType::kLossPrint), 2);

  // disordered 4 mdsave, 2 print, 3 recordload, 2 lossprint
  std::vector<int64_t> vec0{TaskType::kMdSave,     TaskType::kMdSave,     TaskType::kPrint,
                            TaskType::kPrint,      TaskType::kMdSave,     TaskType::kRecordLoad,
                            TaskType::kMdSave,     TaskType::kRecordLoad, TaskType::kLossPrint,
                            TaskType::kRecordLoad, TaskType::kLossPrint};

  // disordered 3 mdsave, 3 print, 2 lossprint
  std::vector<int64_t> vec1{TaskType::kMdSave,    TaskType::kPrint, TaskType::kMdSave,
                            TaskType::kLossPrint, TaskType::kPrint, TaskType::kLossPrint,
                            TaskType::kPrint,     TaskType::kMdSave};

  std::vector<std::vector<int64_t>> machine_task_type{vec0, vec1};
  int64_t base_thrd_id = 16;
  ThrdIdGenerator generator(machine_task_type2thrd_num, base_thrd_id);
  std::vector<std::vector<int64_t>> thrd_ids(machine_task_type.size());
  for (int i = 0; i < machine_task_type.size(); ++i) {
    int64_t machine_id = i;
    std::vector<int64_t> vec = machine_task_type[i];
    for (int64_t task_type : vec) {
      int64_t thrd_id = generator.GenerateThrdId(machine_id, task_type);
      thrd_ids[machine_id].push_back(thrd_id);
    }
  }

  std::vector<int64_t> machine0_thrd_ids{17, 18, 21, 22, 19, 23, 20, 24, 26, 25, 27};
  std::vector<int64_t> machine1_thrd_ids{17, 20, 18, 23, 21, 24, 22, 19};

  CHECK_EQ(thrd_ids[0].size(), machine0_thrd_ids.size());
  CHECK(std::equal(thrd_ids[0].begin(), thrd_ids[0].end(), machine0_thrd_ids.begin()));

  CHECK_EQ(thrd_ids[1].size(), machine1_thrd_ids.size());
  CHECK(std::equal(thrd_ids[1].begin(), thrd_ids[1].end(), machine1_thrd_ids.begin()));
}

TEST(ThrdIdGenerator, ordered_mdsave_num_more_than_config) {
  HashMap<std::pair<int64_t, int64_t>, int32_t> machine_task_type2thrd_num;
  machine_task_type2thrd_num.emplace(std::make_pair(0, TaskType::kMdSave), 4);
  machine_task_type2thrd_num.emplace(std::make_pair(0, TaskType::kPrint), 2);
  machine_task_type2thrd_num.emplace(std::make_pair(0, TaskType::kRecordLoad), 3);
  machine_task_type2thrd_num.emplace(std::make_pair(0, TaskType::kLossPrint), 2);

  machine_task_type2thrd_num.emplace(std::make_pair(1, TaskType::kMdSave), 3);
  machine_task_type2thrd_num.emplace(std::make_pair(1, TaskType::kPrint), 3);
  machine_task_type2thrd_num.emplace(std::make_pair(1, TaskType::kLossPrint), 2);

  // ordered; 4 mdsave, 2 print, 3 recordload, 2 lossprint
  std::vector<int64_t> vec0{// print
                            TaskType::kPrint, TaskType::kPrint,
                            // mdsave
                            TaskType::kMdSave, TaskType::kMdSave, TaskType::kMdSave,
                            TaskType::kMdSave, TaskType::kMdSave, TaskType::kMdSave,
                            // record load
                            TaskType::kRecordLoad, TaskType::kRecordLoad, TaskType::kRecordLoad,
                            // lossprint
                            TaskType::kLossPrint, TaskType::kLossPrint};

  // 3 mdsave, 3 print, 2 lossprint
  std::vector<int64_t> vec1{
      // mdsave
      TaskType::kMdSave, TaskType::kMdSave, TaskType::kMdSave, TaskType::kMdSave, TaskType::kMdSave,
      // print
      TaskType::kPrint, TaskType::kPrint, TaskType::kPrint,
      // lossprint
      TaskType::kLossPrint, TaskType::kLossPrint};

  std::vector<std::vector<int64_t>> machine_task_type{vec0, vec1};
  int64_t base_thrd_id = 16;
  ThrdIdGenerator generator(machine_task_type2thrd_num, base_thrd_id);

  std::vector<std::vector<int64_t>> thrd_ids(machine_task_type.size());
  for (int i = 0; i < machine_task_type.size(); ++i) {
    int64_t machine_id = i;
    std::vector<int64_t> vec = machine_task_type[i];
    for (int64_t task_type : vec) {
      int64_t thrd_id = generator.GenerateThrdId(machine_id, task_type);
      thrd_ids[machine_id].push_back(thrd_id);
    }
  }

  std::vector<int64_t> machine0_thrd_ids{/*Print*/ 17,      18,
                                         /*MdSave*/ 19,     20, 21, 22, 19, 20,
                                         /*RecordLoad*/ 23, 24, 25,
                                         /*LossPrint*/ 26,  27};
  std::vector<int64_t> machine1_thrd_ids{/*MdSave*/ 17,    18, 19, 17, 18,
                                         /*Print*/ 20,     21, 22,
                                         /*LossPrint*/ 23, 24};

  CHECK_EQ(thrd_ids[0].size(), machine0_thrd_ids.size());
  CHECK(std::equal(thrd_ids[0].begin(), thrd_ids[0].end(), machine0_thrd_ids.begin()));

  CHECK_EQ(thrd_ids[1].size(), machine1_thrd_ids.size());
  CHECK(std::equal(thrd_ids[1].begin(), thrd_ids[1].end(), machine1_thrd_ids.begin()));
}

TEST(ThrdIdGenerator, disordered_mdsave_num_more_than_config) {
  HashMap<std::pair<int64_t, int64_t>, int32_t> machine_task_type2thrd_num;
  machine_task_type2thrd_num.emplace(std::make_pair(0, TaskType::kMdSave), 4);
  machine_task_type2thrd_num.emplace(std::make_pair(0, TaskType::kPrint), 2);
  machine_task_type2thrd_num.emplace(std::make_pair(0, TaskType::kRecordLoad), 3);
  machine_task_type2thrd_num.emplace(std::make_pair(0, TaskType::kLossPrint), 2);

  machine_task_type2thrd_num.emplace(std::make_pair(1, TaskType::kMdSave), 3);
  machine_task_type2thrd_num.emplace(std::make_pair(1, TaskType::kPrint), 3);
  machine_task_type2thrd_num.emplace(std::make_pair(1, TaskType::kLossPrint), 2);

  // disordered 7 mdsave, 2 print, 3 recordload, 2 lossprint
  std::vector<int64_t> vec0{TaskType::kPrint,      TaskType::kMdSave,    TaskType::kPrint,
                            TaskType::kMdSave,     TaskType::kMdSave,    TaskType::kMdSave,
                            TaskType::kRecordLoad, TaskType::kMdSave,    TaskType::kMdSave,
                            TaskType::kRecordLoad, TaskType::kLossPrint, TaskType::kMdSave,
                            TaskType::kRecordLoad, TaskType::kLossPrint};

  // disordered 5 mdsave, 3 print, 2 lossprint
  std::vector<int64_t> vec1{TaskType::kMdSave,    TaskType::kPrint,  TaskType::kMdSave,
                            TaskType::kLossPrint, TaskType::kPrint,  TaskType::kMdSave,
                            TaskType::kLossPrint, TaskType::kMdSave, TaskType::kPrint,
                            TaskType::kMdSave};

  std::vector<std::vector<int64_t>> machine_task_type{vec0, vec1};
  int64_t base_thrd_id = 16;
  ThrdIdGenerator generator(machine_task_type2thrd_num, base_thrd_id);

  std::vector<std::vector<int64_t>> thrd_ids(machine_task_type.size());
  for (int i = 0; i < machine_task_type.size(); ++i) {
    int64_t machine_id = i;
    std::vector<int64_t> vec = machine_task_type[i];
    for (int64_t task_type : vec) {
      int64_t thrd_id = generator.GenerateThrdId(machine_id, task_type);
      thrd_ids[machine_id].push_back(thrd_id);
    }
  }

  std::vector<int64_t> machine0_thrd_ids{17, 19, 18, 20, 21, 22, 23, 19, 20, 24, 26, 21, 25, 27};

  std::vector<int64_t> machine1_thrd_ids{17, 20, 18, 23, 21, 19, 24, 17, 22, 18};

  CHECK_EQ(thrd_ids[0].size(), machine0_thrd_ids.size());
  CHECK(std::equal(thrd_ids[0].begin(), thrd_ids[0].end(), machine0_thrd_ids.begin()));

  CHECK_EQ(thrd_ids[1].size(), machine1_thrd_ids.size());
  CHECK(std::equal(thrd_ids[1].begin(), thrd_ids[1].end(), machine1_thrd_ids.begin()));
}

}  // namespace oneflow
