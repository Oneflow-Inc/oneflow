#include "master_session.h"
#include "master_session_interface.h"
#include "master_env.h"

namespace oneflow {
 
class MasterSession : public MasterSessionInterface {
 public:
  MasterSession(const MasterEnv* env){};

 private:
  const MasterEnv* env_;
};

MasterSessionInterface* NewMasterSession(const MasterEnv* env) {
  return new MasterSession(env);
}

}


