#include <iostream>
#include "net/master_session.h"
#include "net/master_session_interface.h"
#include "net/master_env.h"
#include "common_runtime/simple_graph_execution_state.h"


namespace oneflow {
 
class MasterSession : public MasterSessionInterface {
 public:
  MasterSession(const MasterEnv* env){};
  void Create() override;

 private:
  const MasterEnv* env_;
  std::unique_ptr<SimpleGraphExecutionState> execution_state_;
};

void MasterSession::Create() {
  execution_state_.reset(new SimpleGraphExecutionState()); 
  execution_state_->Create();
}

MasterSessionInterface* NewMasterSession(const MasterEnv* env) {
  return new MasterSession(env);
}

}


