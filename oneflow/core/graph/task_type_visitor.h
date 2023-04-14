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
struct TaskTypeVisitor {
  template<typename... Args>
  static auto Visit(TaskType task_type, Args&&... args) {
    switch (task_type) {
      case TaskType::kInvalid: LOG(FATAL) << "invalid task type";
      case TaskType::kNormalForward:
        return DerivedT::VisitNormalForward(std::forward<Args>(args)...);
      case TaskType::kCopyHd: return DerivedT::VisitCopyHd(std::forward<Args>(args)...);
      case TaskType::kCopyCommNet: return DerivedT::VisitCopyCommNet(std::forward<Args>(args)...);
      case TaskType::kDeviceTick: return DerivedT::VisitDeviceTick(std::forward<Args>(args)...);
      case TaskType::kPack: return DerivedT::VisitPack(std::forward<Args>(args)...);
      case TaskType::kUnpack: return DerivedT::VisitUnpack(std::forward<Args>(args)...);
      case TaskType::kRepeat: return DerivedT::VisitRepeat(std::forward<Args>(args)...);
      case TaskType::kAcc: return DerivedT::VisitAcc(std::forward<Args>(args)...);
      case TaskType::kAccCtrlTick: return DerivedT::VisitAccCtrlTick(std::forward<Args>(args)...);
      case TaskType::kSrcSubsetTick:
        return DerivedT::VisitSrcSubsetTick(std::forward<Args>(args)...);
      case TaskType::kDstSubsetTick:
        return DerivedT::VisitDstSubsetTick(std::forward<Args>(args)...);
      case TaskType::kSourceTick: return DerivedT::VisitSourceTick(std::forward<Args>(args)...);
      case TaskType::kTick: return DerivedT::VisitTick(std::forward<Args>(args)...);
      case TaskType::kAccTick: return DerivedT::VisitAccTick(std::forward<Args>(args)...);
      case TaskType::kCase: return DerivedT::VisitCase(std::forward<Args>(args)...);
      case TaskType::kEsac: return DerivedT::VisitEsac(std::forward<Args>(args)...);
      case TaskType::kWaitAndSendIds:
        return DerivedT::VisitWaitAndSendIds(std::forward<Args>(args)...);
      case TaskType::kReentrantLock:
        return DerivedT::VisitReentrantLock(std::forward<Args>(args)...);
      case TaskType::kCallbackNotify:
        return DerivedT::VisitCallbackNotify(std::forward<Args>(args)...);
      case TaskType::kDistributeConcat:
        return DerivedT::VisitDistributeConcat(std::forward<Args>(args)...);
      case TaskType::kDistributeSplit:
        return DerivedT::VisitDistributeSplit(std::forward<Args>(args)...);
      case TaskType::kSliceBoxing: return DerivedT::VisitSliceBoxing(std::forward<Args>(args)...);
      case TaskType::kCollectiveBoxingGeneric:
        return DerivedT::VisitCollectiveBoxingGeneric(std::forward<Args>(args)...);
      case TaskType::kBoxingIdentity:
        return DerivedT::VisitBoxingIdentity(std::forward<Args>(args)...);
      case TaskType::kDecodeH2D: return DerivedT::VisitDecodeH2D(std::forward<Args>(args)...);
      case TaskType::kCollectiveBoxingPack:
        return DerivedT::VisitCollectiveBoxingPack(std::forward<Args>(args)...);
      case TaskType::kCollectiveBoxingUnpack:
        return DerivedT::VisitCollectiveBoxingUnpack(std::forward<Args>(args)...);
      case TaskType::kSspVariableProxy:
        return DerivedT::VisitSspVariableProxy(std::forward<Args>(args)...);
      case TaskType::kBoxingZeros: return DerivedT::VisitBoxingZeros(std::forward<Args>(args)...);
      case TaskType::kCriticalSectionWaitTick:
        return DerivedT::VisitCriticalSectionWaitTick(std::forward<Args>(args)...);
      case TaskType::kNcclSendRecvBoxing:
        return DerivedT::VisitNcclSendRecvBoxing(std::forward<Args>(args)...);
    }
    LOG(FATAL) << "invalid task type";
  }
};

struct IsTransportTaskType final : public TaskTypeVisitor<IsTransportTaskType> {
  static bool VisitCopyHd() { return true; }
  static bool VisitCopyCommNet() { return true; }
  static bool VisitSliceBoxing() { return true; }
  static bool VisitCollectiveBoxingGeneric() { return true; }
  static bool VisitBoxingIdentity() { return true; }
  static bool VisitCollectiveBoxingPack() { return true; }
  static bool VisitCollectiveBoxingUnpack() { return true; }
  static bool VisitNcclSendRecvBoxing() { return true; }
  static bool VisitBoxingZeros() { return true; }

  static bool VisitNormalForward() { return false; }
  static bool VisitDeviceTick() { return false; }
  static bool VisitPack() { return false; }
  static bool VisitUnpack() { return false; }
  static bool VisitRepeat() { return false; }
  static bool VisitAcc() { return false; }
  static bool VisitSrcSubsetTick() { return false; }
  static bool VisitDstSubsetTick() { return false; }
  static bool VisitSourceTick() { return false; }
  static bool VisitTick() { return false; }
  static bool VisitAccTick() { return false; }
  static bool VisitCase() { return false; }
  static bool VisitEsac() { return false; }
  static bool VisitWaitAndSendIds() { return false; }
  static bool VisitReentrantLock() { return false; }
  static bool VisitCallbackNotify() { return false; }
  static bool VisitDistributeConcat() { return false; }
  static bool VisitDistributeSplit() { return false; }
  static bool VisitDecodeH2D() { return false; }
  static bool VisitSspVariableProxy() { return false; }
  static bool VisitCriticalSectionWaitTick() { return false; }
};

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
