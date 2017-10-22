#ifndef ONEFLOW_CORE_CONTROL_CTRL_CLIENT_H_
#define ONEFLOW_CORE_CONTROL_CTRL_CLIENT_H_

#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/comm_network/rdma/connection.h"
#include "oneflow/core/control/ctrl_service.h"

namespace oneflow {

class CtrlClient final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CtrlClient);
  ~CtrlClient() = default;

  OF_SINGLETON(CtrlClient);

  void Barrier(const std::string& barrier_name);
  void Barrier(const std::string& barrier_name, int32_t barrier_num);

  TryLockResult TryLock(const std::string& name);
  void NotifyDone(const std::string& name);
  void WaitUntilDone(const std::string& name);

  void PushPlan(const Plan& plan);
  void ClearPlan();
  void PullPlan(Plan* plan);

  void PushPort(uint16_t port);
  void ClearPort();
  uint16_t PullPort(uint64_t machine_id);

  void PushConnectorInfo();
  void PullConnectorInfo(uint64_t machine_id, ConnectorInfo* info);

 private:
  CtrlClient();
  void LoadServer(const std::string& server_addr, CtrlService::Stub* stub);
  CtrlService::Stub* GetMasterStub() { return stubs_[0].get(); }
  CtrlService::Stub* GetThisStub();
  CtrlService::Stub* GetResponsibleStub(const std::string& key);

  std::vector<std::unique_ptr<CtrlService::Stub>> stubs_;
  HashSet<std::string> done_names_;
};

#define FILE_LINE_STR __FILE__ ":" OF_PP_STRINGIZE(__LINE__)

#define OF_BARRIER() CtrlClient::Singleton()->Barrier(FILE_LINE_STR)

#define OF_CALL_ONCE(name, ...)                                      \
  do {                                                               \
    TryLockResult lock_ret = CtrlClient::Singleton()->TryLock(name); \
    if (lock_ret == TryLockResult::kLocked) {                        \
      __VA_ARGS__;                                                   \
      CtrlClient::Singleton()->NotifyDone(name);                     \
    } else if (lock_ret == TryLockResult::kDone) {                   \
    } else if (lock_ret == TryLockResult::kDoing) {                  \
      CtrlClient::Singleton()->WaitUntilDone(name);                  \
    } else {                                                         \
      UNEXPECTED_RUN();                                              \
    }                                                                \
  } while (0)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CONTROL_CTRL_CLIENT_H_
