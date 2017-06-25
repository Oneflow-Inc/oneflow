#include "oneflow/core/job/id_manager.h"

namespace oneflow {

namespace {

Resource GetResource() {
  Resource ret;
  for(size_t i = 0; i < 10; ++i) {
    Machine* machine = ret.add_machine();
    machine->set_addr("192.168.1."+std::to_string(i));
    machine->set_name("machine_"+std::to_string(i));
    machine->set_port(std::to_string(i+8080));
  }
  ret.set_device_type(DeviceType::kCPU);
  ret.set_device_num_per_machine(8);
  return ret;
}

}

TEST(IDMgr, compile_machine_id_and_name) {
  IDMgr::Singleton().InitFromResource(GetResource());
  ASSERT_EQ(IDMgr::Singleton().MachineID4MachineName("machine_0"), 0);
  ASSERT_EQ(IDMgr::Singleton().MachineID4MachineName("machine_1"), 1);
  ASSERT_EQ(IDMgr::Singleton().MachineID4MachineName("machine_5"), 5);
  ASSERT_EQ(IDMgr::Singleton().MachineName4MachineId(2), "machine_2");
  ASSERT_EQ(IDMgr::Singleton().MachineName4MachineId(3), "machine_3");
  ASSERT_EQ(IDMgr::Singleton().MachineName4MachineId(7), "machine_7");
}

TEST(IDMgr, compile_special_thrd_loc_id) {
  IDMgr::Singleton().InitFromResource(GetResource());
  ASSERT_EQ(IDMgr::Singleton().PersistenceThrdLocId(), 8);
  ASSERT_EQ(IDMgr::Singleton().BoxingThrdLocId(), 9);
  ASSERT_EQ(IDMgr::Singleton().CommNetThrdLocId(), 10);
}

TEST(IDMgr, compile_task_id) {
  IDMgr::Singleton().InitFromResource(GetResource());
  int64_t machine1device2 = (static_cast<int64_t>(1) << (8 + 39))
      + (static_cast<int64_t>(2) << 39);
  ASSERT_EQ(IDMgr::Singleton().NewTaskId(1, 2), machine1device2);
  ASSERT_EQ(IDMgr::Singleton().NewTaskId(1, 2), machine1device2 + 1);
  ASSERT_EQ(IDMgr::Singleton().NewTaskId(1, 2), machine1device2 + 2);
  int64_t machine3device4 = (static_cast<int64_t>(3) << (8 + 39))
      + (static_cast<int64_t>(4) << 39);
  ASSERT_EQ(IDMgr::Singleton().NewTaskId(3, 4), machine3device4);
  ASSERT_EQ(IDMgr::Singleton().NewTaskId(3, 4), machine3device4 + 1);
  ASSERT_EQ(IDMgr::Singleton().NewTaskId(3, 4), machine3device4 + 2);
}

TEST(IDMgr, compile_regst_desc_id) {
  IDMgr::Singleton().InitFromResource(GetResource());
  ASSERT_EQ(IDMgr::Singleton().NewRegstDescId(), 0);
  ASSERT_EQ(IDMgr::Singleton().NewRegstDescId(), 1);
  ASSERT_EQ(IDMgr::Singleton().NewRegstDescId(), 2);
}

TEST(IDMgr, runtime_machine_id) {
  IDMgr::Singleton().InitFromResource(GetResource());
  int64_t actor_id5_machine1device3 = 
      (static_cast<int64_t>(1) << (8 + 39))  // machine_id_1
      + (static_cast<int64_t>(3) << 39)      // device_id_3
      + 5;                                   // actor_id_5
  ASSERT_EQ(IDMgr::Singleton().MachineId4ActorId(actor_id5_machine1device3), 1);
}

TEST(IDMgr, runtime_thrd_loc_id) {
  IDMgr::Singleton().InitFromResource(GetResource());
  int64_t actor_id5_machine1device3 =
      (static_cast<int64_t>(1) << (8 + 39))  // machine_id_1
      + (static_cast<int64_t>(3) << 39)      // device_id_3
      + 5;                                   // actor_id_5
  ASSERT_EQ(IDMgr::Singleton().ThrdLocId4ActorId(actor_id5_machine1device3), 3);
  int64_t actor_id6_machine2device4 =
      (static_cast<int64_t>(2) << (8 + 39))  // machine_id_2
      + (static_cast<int64_t>(4) << 39)      // device_id_4
      + 6;                                   // actor_id_6
  ASSERT_EQ(IDMgr::Singleton().ThrdLocId4ActorId(actor_id6_machine2device4), 4);
}

}// namespace oneflow

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
