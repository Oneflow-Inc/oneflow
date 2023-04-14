#include <glog/logging.h>
#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/graph/collective_boxing_task_node.h"
#include "oneflow/core/graph/nccl_send_recv_boxing_task_node.h"
#include "oneflow/core/graph/copy_task_node.h"
#include "oneflow/core/graph/boxing_zeros_task_node.h"
#include "oneflow/core/graph/slice_boxing_task_node.h"
#include "oneflow/core/graph/collective_boxing_pack_task_node.h"
#include "oneflow/core/graph/collective_boxing_unpack_task_node.h"
#include "oneflow/core/graph/boxing_identity_task_node.h"

namespace oneflow {

template<typename DerivedT>
struct TransportTaskTypeVisitor {
  template<typename... Args>
  static auto Visit(TaskType task_type, Args&&... args) {
    switch (task_type) {
      case TaskType::kInvalid: LOG(FATAL) << "invalid task type";
      case TaskType::kCopyHd: return DerivedT::VisitCopyHd(std::forward<Args>(args)...);
      case TaskType::kCopyCommNet: return DerivedT::VisitCopyCommNet(std::forward<Args>(args)...);
      case TaskType::kSliceBoxing: return DerivedT::VisitSliceBoxing(std::forward<Args>(args)...);
      case TaskType::kCollectiveBoxingGeneric:
        return DerivedT::VisitCollectiveBoxingGeneric(std::forward<Args>(args)...);
      case TaskType::kBoxingIdentity:
        return DerivedT::VisitBoxingIdentity(std::forward<Args>(args)...);
      case TaskType::kNcclSendRecvBoxing:
        return DerivedT::VisitNcclSendRecvBoxing(std::forward<Args>(args)...);
      case TaskType::kBoxingZeros: return DerivedT::VisitBoxingZeros(std::forward<Args>(args)...);
      case TaskType::kCollectiveBoxingPack:
        return DerivedT::VisitCollectiveBoxingPack(std::forward<Args>(args)...);
      case TaskType::kCollectiveBoxingUnpack:
        return DerivedT::VisitCollectiveBoxingUnpack(std::forward<Args>(args)...);
      default: LOG(FATAL) << "invalid task type";
    }
  }
};

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
