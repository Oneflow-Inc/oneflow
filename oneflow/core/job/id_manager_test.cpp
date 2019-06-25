#include "oneflow/core/job/id_manager.h"

namespace oneflow {

namespace {

static const int64_t machine_id_shl = 11 + 21 + 21;
static const int64_t thread_id_shl = 21 + 21;
static const int64_t local_work_stream_shl = 21;

Resource GetResource() {
  Resource ret;
  for (size_t i = 0; i < 10; ++i) {
    Machine* machine = ret.add_machine();
    machine->set_addr("192.168.1." + std::to_string(i));
    machine->set_name("machine_" + std::to_string(i));
    machine->set_port(i + 8080);
  }
  ret.set_gpu_device_num(8);
  ret.set_cpu_device_num(5);
  ret.set_comm_net_worker_num(4);
  ret.set_persistence_worker_num(30);
  return ret;
}

void New() {
  JobDescProto proto;
  *proto.mutable_resource() = GetResource();
  TODO();
  // Global<JobDesc>::New(proto);
  Global<IDMgr>::New();
}

void Delete() {
  Global<IDMgr>::Delete();
  TODO();
  // Global<JobDesc>::Delete();
}

}  // namespace

TEST(IDMgr, compile_machine_id_and_name) {
  New();
  ASSERT_EQ(Global<IDMgr>::Get()->MachineID4MachineName("machine_0"), 0);
  ASSERT_EQ(Global<IDMgr>::Get()->MachineID4MachineName("machine_1"), 1);
  ASSERT_EQ(Global<IDMgr>::Get()->MachineID4MachineName("machine_5"), 5);
  ASSERT_EQ(Global<IDMgr>::Get()->MachineName4MachineId(2), "machine_2");
  ASSERT_EQ(Global<IDMgr>::Get()->MachineName4MachineId(3), "machine_3");
  ASSERT_EQ(Global<IDMgr>::Get()->MachineName4MachineId(7), "machine_7");
  Delete();
}

TEST(IDMgr, compile_special_thrd_id) {
  New();
  ASSERT_EQ(Global<IDMgr>::Get()->GetPersistenceThrdId(1), 8 + 5 + 1);
  ASSERT_EQ(Global<IDMgr>::Get()->CommNetThrdId(), 8 + 5 + 30);
  Delete();
}

TEST(IDMgr, compile_task_id) {
  New();
  int64_t machine1thrd2 =
      (static_cast<int64_t>(1) << machine_id_shl) + (static_cast<int64_t>(2) << thread_id_shl);
  int64_t machine3thrd4 =
      (static_cast<int64_t>(3) << machine_id_shl) + (static_cast<int64_t>(4) << thread_id_shl);
  int64_t local_work_stream1 = (static_cast<int64_t>(1) << local_work_stream_shl);
  int64_t local_work_stream3 = (static_cast<int64_t>(3) << local_work_stream_shl);
  ASSERT_EQ(Global<IDMgr>::Get()->NewTaskId(1, 2, 0), machine1thrd2 | 0 | 0);
  ASSERT_EQ(Global<IDMgr>::Get()->NewTaskId(1, 2, 1), machine1thrd2 | local_work_stream1 | 1);
  ASSERT_EQ(Global<IDMgr>::Get()->NewTaskId(1, 2, 1), machine1thrd2 | local_work_stream1 | 2);
  ASSERT_EQ(Global<IDMgr>::Get()->NewTaskId(3, 4, 1), machine3thrd4 | local_work_stream1 | 0);
  ASSERT_EQ(Global<IDMgr>::Get()->NewTaskId(3, 4, 1), machine3thrd4 | local_work_stream1 | 1);
  ASSERT_EQ(Global<IDMgr>::Get()->NewTaskId(3, 4, 3), machine3thrd4 | local_work_stream3 | 2);
  Delete();
}

TEST(IDMgr, compile_regst_desc_id) {
  New();
  ASSERT_EQ(Global<IDMgr>::Get()->NewRegstDescId(), 0);
  ASSERT_EQ(Global<IDMgr>::Get()->NewRegstDescId(), 1);
  ASSERT_EQ(Global<IDMgr>::Get()->NewRegstDescId(), 2);
  Delete();
}

TEST(IDMgr, runtime_machine_id) {
  New();
  int64_t actor_id5_machine1thrd3 =
      (static_cast<int64_t>(1) << machine_id_shl)           // machine_id_1
      + (static_cast<int64_t>(3) << thread_id_shl)          // thrd_id_3
      + (static_cast<int64_t>(1) << local_work_stream_shl)  // work_stream_id_1
      + 5;                                                  // actor_id_5
  ASSERT_EQ(Global<IDMgr>::Get()->MachineId4ActorId(actor_id5_machine1thrd3), 1);
  Delete();
}

TEST(IDMgr, runtime_thrd_id) {
  New();
  int64_t actor_id5_machine1thrd3 = (static_cast<int64_t>(1) << machine_id_shl)   // machine_id_1
                                    + (static_cast<int64_t>(3) << thread_id_shl)  // thrd_id_3
                                    // work_stream_id_0
                                    + 5;  // actor_id_5
  ASSERT_EQ(Global<IDMgr>::Get()->ThrdId4ActorId(actor_id5_machine1thrd3), 3);
  int64_t actor_id6_machine2thrd4 =
      (static_cast<int64_t>(2) << machine_id_shl)             // machine_id_2
      + (static_cast<int64_t>(4) << thread_id_shl)            // thrd_id_4
      + (static_cast<int64_t>(101) << local_work_stream_shl)  // work_stream_id_101
      + 6;                                                    // actor_id_6
  ASSERT_EQ(Global<IDMgr>::Get()->ThrdId4ActorId(actor_id6_machine2thrd4), 4);
  Delete();
}

}  // namespace oneflow
