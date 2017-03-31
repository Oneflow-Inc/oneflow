#ifndef MASTER_SESSION_H_
#define MASTER_SESSION_H_

#include <memory>

namespace oneflow {

class MasterEnv;
class MasterSessionInterface;

MasterSessionInterface* NewMasterSession(const MasterEnv* env);

}

#endif
