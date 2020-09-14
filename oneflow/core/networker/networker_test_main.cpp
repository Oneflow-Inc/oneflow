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
#include "oneflow/core/networker/networker.h"

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
  Global<Networker>::Delete();
  Global<EpollCommNet>::Delete();
  Global<EnvGlobalObjectsScope>::Delete();
}

Maybe<void> TestNetworkerOn2Machine(const std::string& first_machine_ip,
                                    const std::string& second_machine_ip) {
  CHECK_ISNULL_OR_RETURN(Global<EnvGlobalObjectsScope>::Get());
  // Global<T>::New is not allowed to be called here
  // because glog is not constructed yet and LOG(INFO) has bad bahavior
  Global<EnvGlobalObjectsScope>::SetAllocated(new EnvGlobalObjectsScope());
  JUST(
      Global<EnvGlobalObjectsScope>::Get()->Init(GetEnvProto(first_machine_ip, second_machine_ip)));

  // do networker test
  Global<EpollCommNet>::New();
  Global<Networker>::New();

  std::cout << "New All Global" << std::endl;
  BlockingCounter bc(1);
  OF_BARRIER();

  int64_t this_machine_id = Global<MachineCtx>::Get()->this_machine_id();
  if (this_machine_id == 0) {
    std::cout << "I'm first machine!" << std::endl;
    void* first_send_ptr = malloc(1024);
    int32_t* data = static_cast<int32_t*>(first_send_ptr);
    for (int i = 0; i < 1024 / 4; ++i) { *(data + i) = i; }
    std::function<void()> func = [first_send_ptr, &bc]() {
      std::cout << "Yes! I have send 1024 bytes to machine 1" << std::endl;
      free(first_send_ptr);
      DeleteAll();
      bc.Decrease();
    };
    Global<Networker>::Get()->Send(23330, 1, first_send_ptr, 1024, func);
    std::cout << "First send post!" << std::endl;
  } else if (this_machine_id == 1) {
    std::cout << "I'm second machine!" << std::endl;
    void* first_receive_ptr = malloc(1024);
    std::function<void()> func = [first_receive_ptr, &bc]() {
      int32_t* data = static_cast<int32_t*>(first_receive_ptr);
      for (int i = 0; i < 1024 / 4; ++i) { CHECK_EQ(*(data + i), i); }
      std::cout << "Yes! I have recv 1024 bytes from machine 0" << std::endl;
      free(first_receive_ptr);
      DeleteAll();
      bc.Decrease();
    };
    Global<Networker>::Get()->Receive(23330, 0, first_receive_ptr, 1024, func);
    std::cout << "First recv post!" << std::endl;
  } else {
    UNIMPLEMENTED();
  }
  std::cout << "wait for callback..." << std::endl;
  bc.WaitUntilCntEqualZero();
  std::cout << "All Done!" << std::endl;

  return Maybe<void>::Ok();
}

}  // namespace

}  // namespace oneflow

/*
 * Try run this test exe by :
 *     ./oneflow_test_networker first_machine_ip="192.168.1.15" \
 *          second_machine_ip = "192.168.1.16"
 */
DEFINE_string(first_machine_ip, "192.168.1.15", "IP address for first machine");
DEFINE_string(second_machine_ip, "192.168.1.16", "IP address for second machine");

int main(int argc, char* argv[]) {
  using namespace oneflow;
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  CHECK_JUST(TestNetworkerOn2Machine(FLAGS_first_machine_ip, FLAGS_second_machine_ip));
  return 0;
}
