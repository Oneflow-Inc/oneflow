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
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/oneflow.h"
#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/core/job/session_global_objects_scope.h"
#include "oneflow/core/job/env.pb.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/ctrl_server.h"
#include "oneflow/core/control/ctrl_bootstrap.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/transport/transport.h"

#include <chrono>
#include <iomanip>

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
      Global<Transport>::Get()->Receive(token, 1, ptr, size + 77, [ptr, i, size, &bc]() {
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
      Global<Transport>::Get()->Receive(token, 0, ptr, size + 66, [ptr, i, size, &bc]() {
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
  std::cout << "Test for correctness. Start. \nEach machine will send and receive 100 messages (50 "
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

  for (int i = 0; i < test_num; ++i) {
    malloc_size_list.at(i) = 16 << (i % 20);
    total_bytes += malloc_size_list.at(i);
    // malloc data
    malloc_ptr_list.at(i) = malloc(malloc_size_list.at(i) + 100);
  }
  total_mib = (total_bytes * 1.0 / 1000000.0);

  int64_t this_machine_id = GlobalProcessCtx::Rank();
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
  std::cout << "Test for correctness. Done.\n\n";
}

void TestCorrectnessOnLocalMachine() {
  std::cout << "Test for local send/recv transport correctness. Start. \n"
               "Machine will send and receive 100 messages alternately. "
               "The first address and the last address of each "
               "transport are written with data for correctness verification.\n";
  uint64_t first_token = 3456;
  size_t test_num = 100;
  uint64_t total_bytes = 0;
  double total_mib = -1;
  std::vector<uint64_t> send_malloc_size_list(test_num);
  std::vector<void*> send_malloc_ptr_list(test_num);
  std::vector<uint64_t> recv_malloc_size_list(test_num);
  std::vector<void*> recv_malloc_ptr_list(test_num);

  for (int i = 0; i < test_num; ++i) {
    send_malloc_size_list.at(i) = 16 << (i % 20);
    recv_malloc_size_list.at(i) = send_malloc_size_list.at(i) + 100;
    total_bytes += send_malloc_size_list.at(i);
    // malloc data
    send_malloc_ptr_list.at(i) = malloc(send_malloc_size_list.at(i));
    recv_malloc_ptr_list.at(i) = malloc(recv_malloc_size_list.at(i));
  }
  total_mib = (total_bytes * 1.0 / 1000000.0);

  int64_t this_machine_id = GlobalProcessCtx::Rank();

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  BlockingCounter bc(test_num * 2);
  for (int i = 0; i < test_num; ++i) {
    // Send
    void* ptr = send_malloc_ptr_list.at(i);
    size_t size = send_malloc_size_list.at(i);
    uint64_t token = first_token + i;
    int32_t* data = static_cast<int32_t*>(ptr);
    *data = i;
    *(data + (size / 4) - 1) = i + 10;
    Global<Transport>::Get()->Send(token, this_machine_id, ptr, size, [&bc]() { bc.Decrease(); });
  }

  for (int i = test_num - 1; i >= 0; --i) {
    void* ptr = recv_malloc_ptr_list.at(i);
    size_t size = recv_malloc_size_list.at(i);
    CHECK(size > 100);
    uint64_t token = first_token + i;
    // Recv
    Global<Transport>::Get()->Receive(token, this_machine_id, ptr, size, [ptr, i, size, &bc]() {
      int32_t* data = static_cast<int32_t*>(ptr);
      CHECK_EQ(*data, i);
      CHECK_EQ(*(data + ((size - 100) / 4) - 1), i + 10);
      bc.Decrease();
    });
  }
  bc.WaitUntilCntEqualZero();
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  double duration_sec =
      std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0;
  std::cout << "the latency is : " << duration_sec
            << " s, the throughput is : " << total_mib / duration_sec << " MiB/s \n";

  // free ptr
  for (int i = 0; i < test_num; ++i) {
    CHECK(send_malloc_ptr_list.at(i) != nullptr);
    free(send_malloc_ptr_list.at(i));
    CHECK(recv_malloc_ptr_list.at(i) != nullptr);
    free(recv_malloc_ptr_list.at(i));
  }
  std::cout << "Test for local send/recv transport correctness. Done.\n\n";
}

void TestThroughputWithBytes(uint64_t bytes, uint64_t first_token) {
  int32_t total_iteration = 1000;
  int64_t this_machine_id = GlobalProcessCtx::Rank();
  std::vector<std::chrono::steady_clock::time_point> time_points(1010,
                                                                 std::chrono::steady_clock::now());
  std::size_t size = bytes;
  void* ptr = malloc(size);
  BlockingCounter bc(total_iteration);
  time_points.at(0) = std::chrono::steady_clock::now();
  if (this_machine_id == 0) {
    for (int i = 0; i < total_iteration; ++i) {
      Global<Transport>::Get()->Send(first_token + i, 1, ptr, size, [&bc, &time_points, i]() {
        time_points.at(i + 1) = std::chrono::steady_clock::now();
        bc.Decrease();
      });
    }
  } else if (this_machine_id == 1) {
    for (int i = 0; i < total_iteration; ++i) {
      Global<Transport>::Get()->Receive(first_token + i, 0, ptr, size, [&bc, &time_points, i]() {
        time_points.at(i + 1) = std::chrono::steady_clock::now();
        bc.Decrease();
      });
    }
  } else {
    UNIMPLEMENTED();
  }
  bc.WaitUntilCntEqualZero();

  double throughput_peak = 0;
  double total_bytes = bytes * 1000;
  for (int i = 1; i <= total_iteration; ++i) {
    double duration_micro_sec = std::chrono::duration_cast<std::chrono::microseconds>(
                                    time_points.at(i) - time_points.at(i - 1))
                                    .count();
    throughput_peak = std::max(throughput_peak, bytes * 1.0 / duration_micro_sec);  // MiB/s
  }
  double throughput_average = total_bytes
                              / (std::chrono::duration_cast<std::chrono::microseconds>(
                                     time_points.at(1000) - time_points.at(0))
                                     .count());
  std::cout << std::setw(25) << std::left << bytes << std::setw(25) << std::left << 1000
            << std::setw(25) << std::left << throughput_peak << std::setw(25) << std::left
            << throughput_average << std::endl;
}

void TestThroughput() {
  std::cout << "Test for throughput. Start.\n";
  std::cout << "-------------------------------------------------------------------------------\n";
  std::cout << std::setw(25) << std::left << "#bytes" << std::setw(25) << std::left << "#iterations"
            << std::setw(25) << std::left << "#throughput peek[MiB/s]" << std::setw(25) << std::left
            << "#throughput average[MiB/s]" << std::endl;
  uint64_t bytes = 2;
  for (int i = 0; i < 23; ++i) { TestThroughputWithBytes(bytes << i, 10000 * (i + 1)); }
  std::cout << "-------------------------------------------------------------------------------\n";
  std::cout << "Test for throughput. Done.\n\n";
}

Maybe<void> TestTransportOn2Machine(const std::string& first_machine_ip,
                                    const std::string& second_machine_ip, int32_t ctrl_port) {
  EnvProto env_proto = GetEnvProto(first_machine_ip, second_machine_ip, ctrl_port);
  Global<EnvDesc>::New(env_proto);
  Global<CtrlServer>::New();
  Global<ProcessCtx>::New();
  JUST(HostListCtrlBootstrap(*Global<EnvDesc>::Get())
           .InitProcessCtx(Global<CtrlServer>::Get()->port(), Global<ProcessCtx>::Get()));
  Global<CtrlClient>::New(*Global<ProcessCtx>::Get());
  Global<ResourceDesc, ForEnv>::New(GetResource());
  Global<ResourceDesc, ForSession>::New(GetResource());

  // do transport test
  // The Global<EpollCommNet> must new first before Global<Transport> new.
  std::cout << "New All Global" << std::endl;
  Global<EpollCommNet>::New();
  Global<Transport>::New();

  // OF_ENV_BARRIER Must call before test,
  // to ensure that the Global<Transport> on each machine is created
  OF_ENV_BARRIER();

  // Test for correctness
  // Each machine will send and receive 100 messages (50 send and 50 recv) alternately.
  // The first address and the last address of each transport
  // are written with data for correctness verification.
  TestCorrectness();
  TestCorrectnessOnLocalMachine();

  TestThroughput();

  OF_ENV_BARRIER();
  std::cout << "Deleting all global..." << std::endl;
  Global<Transport>::Delete();
  Global<EpollCommNet>::Delete();
  Global<ResourceDesc, ForSession>::Delete();
  Global<ResourceDesc, ForEnv>::Delete();
  Global<CtrlClient>::Delete();
  Global<ProcessCtx>::Delete();
  Global<CtrlServer>::Delete();
  Global<EnvDesc>::Delete();
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
