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
  static Maybe<TransportTaskNode*> VisitCollectiveBoxingGeneric() { return new CollectiveBoxingGenericTaskNode(); }
  static Maybe<TransportTaskNode*> VisitBoxingIdentity() { return new BoxingIdentityTaskNode(); }
  static Maybe<TransportTaskNode*> VisitCollectiveBoxingPack() { return new CollectiveBoxingPackTaskNode(); }
  static Maybe<TransportTaskNode*> VisitCollectiveBoxingUnpack() { return new CollectiveBoxingUnpackTaskNode(); }
  static Maybe<TransportTaskNode*> VisitBoxingZeros() { return new BoxingZerosTaskNode(); }
  static Maybe<TransportTaskNode*> VisitNcclSendRecvBoxing() { return new NcclSendRecvBoxing(); }
};

}
