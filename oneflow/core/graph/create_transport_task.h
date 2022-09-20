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
#include "oneflow/core/graph/task_type.h"
#include "oneflow/core/graph/collective_boxing_task_node.h"
#include "oneflow/core/graph/nccl_send_recv_boxing_task_node.h"
#include "oneflow/core/graph/copy_task_node.h"
#include "oneflow/core/graph/boxing_zeros_task_node.h"
#include "oneflow/core/graph/slice_boxing_task_node.h"
#include "oneflow/core/graph/collective_boxing_pack_task_node.h"
#include "oneflow/core/graph/collective_boxing_unpack_task_node.h"
#include "oneflow/core/graph/boxing_identity_task_node.h"

namespace oneflow {

struct CreateTransportTask final : public TransportTaskTypeVisitor<CreateTransportTask> {
  static Maybe<TransportTaskNode*> VisitCopyHd() { return new CopyHdTaskNode(); }
  static Maybe<TransportTaskNode*> VisitCopyCommNet() { return new CopyCommNetTaskNode(); }
  static Maybe<TransportTaskNode*> VisitSliceBoxing() { return new SliceBoxingTaskNode(); }
  static Maybe<TransportTaskNode*> VisitCollectiveBoxingGeneric() {
    return new CollectiveBoxingGenericTaskNode();
  }
  static Maybe<TransportTaskNode*> VisitBoxingIdentity() { return new BoxingIdentityTaskNode(); }
  static Maybe<TransportTaskNode*> VisitCollectiveBoxingPack() {
    return new CollectiveBoxingPackTaskNode();
  }
  static Maybe<TransportTaskNode*> VisitCollectiveBoxingUnpack() {
    return new CollectiveBoxingUnpackTaskNode();
  }
  static Maybe<TransportTaskNode*> VisitBoxingZeros() { return new BoxingZerosTaskNode(); }
  static Maybe<TransportTaskNode*> VisitNcclSendRecvBoxing() {
    return new NcclSendRecvBoxingTaskNode();
  }
};

}  // namespace oneflow
