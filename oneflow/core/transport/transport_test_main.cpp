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
#include "oneflow/core/job/cluster.h"
#include "oneflow/core/control/cluster_control.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/ctrl_server.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/transport/transport.h"

namespace oneflow {

namespace {

EnvProto GetEnvProto(const std::string& first_machine_ip, const std::string& second_machine_ip) {
  EnvProto ret;
  auto* machine0 = ret.add_machine();
  machine0->set_id(0);
  machine0->set_addr(first_machine_ip);
  auto* machine1 = ret.add_machine();
  machine1->set_id(1);
  machine1->set_addr(second_machine_ip);
  ret.set_ctrl_port(12144);
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

Maybe<void> TestTransportOn2Machine(const std::string& first_machine_ip,
                                    const std::string& second_machine_ip) {
  CHECK_ISNULL_OR_RETURN(Global<EnvGlobalObjectsScope>::Get());
  // Global<T>::New is not allowed to be called here
  // because glog is not constructed yet and LOG(INFO) has bad bahavior
  Global<EnvGlobalObjectsScope>::SetAllocated(new EnvGlobalObjectsScope());
  JUST(
      Global<EnvGlobalObjectsScope>::Get()->Init(GetEnvProto(first_machine_ip, second_machine_ip)));

  // do transport test
  Global<EpollCommNet>::New();
  Global<Transport>::New();

  std::cout << "New All Global" << std::endl;
  OF_BARRIER();

  void* malloc_ptr = nullptr;

  int64_t this_machine_id = Global<MachineCtx>::Get()->this_machine_id();
  if (this_machine_id == 0) {
    // malloc data
    std::cout << "I'm first machine!" << std::endl;
    malloc_ptr = malloc(1024);
    int32_t* data = static_cast<int32_t*>(malloc_ptr);
    for (int i = 0; i < 1024 / 4; ++i) { *(data + i) = i; }

    // send
    BlockingCounter bc(1);
    Global<Transport>::Get()->Send(23330, 1, malloc_ptr, 1024, [malloc_ptr, &bc]() {
      std::cout << "Yes! I have send 1024 bytes to machine 1" << std::endl;
      bc.Decrease();
    });
    std::cout << "First send post!" << std::endl;
    std::cout << "wait for callback..." << std::endl;
    bc.WaitUntilCntEqualZero();
    std::cout << "First send done!" << std::endl;
  } else if (this_machine_id == 1) {
    std::cout << "I'm second machine!" << std::endl;
    malloc_ptr = malloc(1024);

    // receive
    BlockingCounter bc(1);
    Global<Transport>::Get()->Receive(23330, 0, malloc_ptr, 1024, [malloc_ptr, &bc]() {
      int32_t* data = static_cast<int32_t*>(malloc_ptr);
      for (int i = 0; i < 1024 / 4; ++i) { CHECK_EQ(*(data + i), i); }
      std::cout << "Yes! I have recv 1024 bytes from machine 0" << std::endl;
      bc.Decrease();
    });
    std::cout << "First recv post!" << std::endl;
    std::cout << "wait for callback..." << std::endl;
    bc.WaitUntilCntEqualZero();
    std::cout << "First recv done!" << std::endl;
  } else {
    UNIMPLEMENTED();
  }
  free(malloc_ptr);
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
 *          second_machine_ip = "192.168.1.16"
 */
DEFINE_string(first_machine_ip, "192.168.1.15", "IP address for first machine");
DEFINE_string(second_machine_ip, "192.168.1.16", "IP address for second machine");

int main(int argc, char* argv[]) {
  using namespace oneflow;
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  CHECK_JUST(TestTransportOn2Machine(FLAGS_first_machine_ip, FLAGS_second_machine_ip));
  return 0;
}
