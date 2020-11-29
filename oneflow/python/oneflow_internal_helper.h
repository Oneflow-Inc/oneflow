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
#include <iostream>
#include <google/protobuf/text_format.h>
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/operator/op_attribute.pb.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/env.pb.h"
#include "oneflow/core/job/oneflow.h"
#include "oneflow/core/job/foreign_job_instance.h"
#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/core/job/session_global_objects_scope.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/oneflow.h"
#include "oneflow/core/job/runtime_job_descs.h"
#include "oneflow/core/job/cluster_instruction.pb.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/cluster_instruction.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/foreign_watcher.h"
#include "oneflow/core/job/foreign_callback.h"
#include "oneflow/core/job/cluster.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/framework/config_def.h"
#include "oneflow/core/framework/load_library.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/vm/instruction.pb.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/id_util.h"
#include "oneflow/core/eager/eager_oneflow.h"
#include "oneflow/core/eager/eager_symbol_storage.h"

namespace oneflow {

Maybe<void> RegisterWatcherOnlyOnce(ForeignWatcher* watcher) {
  CHECK_ISNULL_OR_RETURN(Global<ForeignWatcher>::Get()) << "foreign watcher registered";
  Global<ForeignWatcher>::SetAllocated(watcher);
  return Maybe<void>::Ok();
}

Maybe<void> LaunchJob(const std::shared_ptr<oneflow::ForeignJobInstance>& cb) {
  CHECK_OR_RETURN(Global<MachineCtx>::Get()->IsThisMachineMaster());
  CHECK_NOTNULL_OR_RETURN(Global<Oneflow>::Get());
  const auto& job_name = cb->job_name();
  auto* buffer_mgr = Global<BufferMgr<std::shared_ptr<ForeignJobInstance>>>::Get();
  int64_t job_id = Global<JobName2JobId>::Get()->at(job_name);
  if (IsPullJob(job_name, *Global<InterUserJobInfo>::Get())) {
    buffer_mgr->Get(GetForeignOutputBufferName(job_name))->Send(cb);
  }
  if (IsPushJob(job_name, *Global<InterUserJobInfo>::Get())) {
    buffer_mgr->Get(GetForeignInputBufferName(job_name))->Send(cb);
  }
  buffer_mgr->Get(GetCallbackNotifierBufferName(job_name))->Send(cb);
  Global<BufferMgr<int64_t>>::Get()->Get(kBufferNameGlobalWaitJobId)->Send(job_id);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
