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
#include "oneflow/core/job/oneflow.h"
#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/core/job/session_global_objects_scope.h"
#include "oneflow/core/job/env.pb.h"
#include "oneflow/core/job/cluster.h"
#include "oneflow/core/job/cluster_instruction.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/ctrl_server.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"

namespace oneflow {

namespace {

Maybe<void> Run(const std::string& env_proto_filepath) {
  EnvProto env_proto;
  ParseProtoFromTextFile(env_proto_filepath, &env_proto);
  CHECK_ISNULL_OR_RETURN(Global<EnvGlobalObjectsScope>::Get());
  // Global<T>::New is not allowed to be called here
  // because glog is not constructed yet and LOG(INFO) has bad bahavior
  Global<EnvGlobalObjectsScope>::SetAllocated(new EnvGlobalObjectsScope());
  JUST(Global<EnvGlobalObjectsScope>::Get()->Init(env_proto));
  JUST(Cluster::WorkerLoop());
  return Maybe<void>::Ok();
}

}  // namespace

}  // namespace oneflow

DEFINE_string(env_proto, "", "EnvProto file path");

int main(int argc, char* argv[]) {
  using namespace oneflow;
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  CHECK_JUST(Run(FLAGS_env_proto));
  return 0;
}
