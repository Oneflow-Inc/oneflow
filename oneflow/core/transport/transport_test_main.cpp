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
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/job/oneflow.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/core/job/session_global_objects_scope.h"
#include "oneflow/core/job/env.pb.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/ctrl_server.h"
#include "oneflow/core/transport/transport.h"

#include <chrono>

namespace oneflow {

namespace {

EnvProto GetEnvProto(const std::string& first_machine_ip, const std::string& second_machine_ip,
                     int32_t ctrl_port) {
  EnvProto ret;
  auto* machine0 = ret.add_machine();
  machine0->set_id(0);
  machine0->set_addr(first_machine_ip);
  auto* machine1 = ret.add_machine();
  machine1->set_id(1);
  machine1->set_addr(second_machine_ip);
  ret.set_ctrl_port(ctrl_port);
  return ret;
}

Resource GetResource() {
  Resource ret;
  ret.set_machine_num(2);
  ret.set_gpu_device_num(0);
  ret.set_cpu_device_num(1);
  ret.set_comm_net_worker_num(1);
  return ret;
}

void DeleteAll() {
  Global<Transport>::Delete();
  Global<EpollCommNet>::Delete();
  Global<EnvGlobalObjectsScope>::Delete();
}

void HandlerOfFirstMachine(uint64_t first_token, size_t test_num,
                           const std::vector<uint64_t>& malloc_size_list,
                           const std::vector<void*>& malloc_ptr_list) {
  BlockingCounter bc(test_num);
  for (int i = 0; i < test_num; ++i) {
    void* ptr = malloc_ptr_list.at(i);
    size_t size = malloc_size_list.at(i);
    uint64_t token = first_token + i;
    if (i % 2 == 0) {
      // Send
      int32_t* data = static_cast<int32_t*>(ptr);
      *data = i;
      *(data + (size / 4) - 1) = i + 10;
      Global<Transport>::Get()->Send(token, 1, ptr, size, [&bc]() { bc.Decrease(); });
    } else {
      // Recv
      Global<Transport>::Get()->Receive(token, 1, ptr, size, [ptr, i, size, &bc]() {
        int32_t* data = static_cast<int32_t*>(ptr);
        CHECK_EQ(*data, i);
        CHECK_EQ(*(data + (size / 4) - 1), i + 20);
        bc.Decrease();
      });
    }
  }
  bc.WaitUntilCntEqualZero();
}

void HandlerOfSecondMachine(uint64_t first_token, size_t test_num,
                            const std::vector<uint64_t>& malloc_size_list,
                            const std::vector<void*>& malloc_ptr_list) {
  BlockingCounter bc(test_num);
  for (int i = 0; i < test_num; ++i) {
    void* ptr = malloc_ptr_list.at(i);
    size_t size = malloc_size_list.at(i);
    uint64_t token = first_token + i;
    if (i % 2 == 0) {
      // Recv
      Global<Transport>::Get()->Receive(token, 0, ptr, size, [ptr, i, size, &bc]() {
        int32_t* data = static_cast<int32_t*>(ptr);
        CHECK_EQ(*data, i);
        CHECK_EQ(*(data + (size / 4) - 1), i + 10);
        bc.Decrease();
      });

    } else {
      // Send
      int32_t* data = static_cast<int32_t*>(ptr);
      *data = i;
      *(data + (size / 4) - 1) = i + 20;
      Global<Transport>::Get()->Send(token, 0, ptr, size, [&bc]() { bc.Decrease(); });
    }
  }
  bc.WaitUntilCntEqualZero();
}

void TestCorrectness() {
  std::cout << "Test for correctness. Begin. \nEach machine will send and receive 100 messages (50 "
               "send and 50 recv) alternately. The first address and the last address of each "
               "transport are written with data for correctness verification.\n";
  uint64_t first_token = 2333;
  size_t test_num = 100;
  // uint64_t min_bytes = 16;
  // uint64_t max_bytes = 16 << 20;  // 16 MB
  uint64_t total_bytes = 0;
  double total_mib = -1;
  std::vector<uint64_t> malloc_size_list(test_num);
  std::vector<void*> malloc_ptr_list(test_num);

  // std::cout << "malloc list = [";
  for (int i = 0; i < test_num; ++i) {
    malloc_size_list.at(i) = 16 << (i % 20);
    // std::cout << malloc_size_list.at(i) << ",";
    total_bytes += malloc_size_list.at(i);
    // malloc data
    malloc_ptr_list.at(i) = malloc(malloc_size_list.at(i));
  }
  // std::cout << "]\n";
  total_mib = (total_bytes * 1.0 / 1000000.0);
  // std::cout << "totol transport bytes is " << total_mib << " MiB\n";

  int64_t this_machine_id = Global<MachineCtx>::Get()->this_machine_id();
  if (this_machine_id == 0) {
    std::cout << "I'm first machine!" << std::endl;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    HandlerOfFirstMachine(first_token, test_num, malloc_size_list, malloc_ptr_list);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double duration_sec =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0;
    std::cout << "the latency is : " << duration_sec
              << " s, the throughput is : " << total_mib / duration_sec << " MiB/s \n";
  } else if (this_machine_id == 1) {
    std::cout << "I'm second machine!" << std::endl;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    HandlerOfSecondMachine(first_token, test_num, malloc_size_list, malloc_ptr_list);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double duration_sec =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0;
    std::cout << "the latency is : " << duration_sec
              << " s, the throughput is : " << total_mib / duration_sec << " MiB/s \n";
  } else {
    UNIMPLEMENTED();
  }
  // free ptr
  for (int i = 0; i < test_num; ++i) {
    CHECK(malloc_ptr_list.at(i) != nullptr);
    free(malloc_ptr_list.at(i));
  }
  std::cout << "Test for correctness. Done.\n";
}

Maybe<void> TestTransportOn2Machine(const std::string& first_machine_ip,
                                    const std::string& second_machine_ip, int32_t ctrl_port) {
  CHECK_ISNULL_OR_RETURN(Global<EnvGlobalObjectsScope>::Get());
  // Global<T>::New is not allowed to be called here
  // because glog is not constructed yet and LOG(INFO) has bad bahavior
  Global<EnvGlobalObjectsScope>::SetAllocated(new EnvGlobalObjectsScope());
  JUST(Global<EnvGlobalObjectsScope>::Get()->Init(
      GetEnvProto(first_machine_ip, second_machine_ip, ctrl_port)));

  // do transport test
  // The Global<EpollCommNet> must new first before Global<Transport> new.
  std::cout << "New All Global" << std::endl;
  Global<EpollCommNet>::New();
  Global<Transport>::New();

  // OF_BARRIER Must call before test,
  // to ensure that the Global<Transport> on each machine is created
  OF_BARRIER();

  // Test for correctness
  // Each machine will send and receive 100 messages (50 send and 50 recv) alternately.
  // The first address and the last address of each transport
  // are written with data for correctness verification.
  TestCorrectness();

  std::cout << "Deleting all global..." << std::endl;
  DeleteAll();
  std::cout << "All Done!" << std::endl;
  return Maybe<void>::Ok();
}

}  // namespace

}  // namespace oneflow

/*
 * Try run this test exe by :
 *     ./oneflow_test_transport first_machine_ip="192.168.1.15" \
 *          second_machine_ip = "192.168.1.16" ctrl_port=12143
 */
DEFINE_string(first_machine_ip, "192.168.1.15", "IP address for first machine.");
DEFINE_string(second_machine_ip, "192.168.1.16", "IP address for second machine.");
DEFINE_int32(ctrl_port, 12143, "the control port for init CtrlServer/Client.");

int main(int argc, char* argv[]) {
  using namespace oneflow;
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  CHECK_JUST(
      TestTransportOn2Machine(FLAGS_first_machine_ip, FLAGS_second_machine_ip, FLAGS_ctrl_port));
  return 0;
}
