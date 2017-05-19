
#include "network/network.h"

namespace oneflow {

struct NetworkMessage;
struct MemoryDescriptor;
struct NetworkMemory;

class GrpcWorker : Network {
  public:
    GrpcWorker();
    ~GrpcWorker();
    bool Send(const NetworkMessage& msg);
    void Read(MemoryDescriptor* src, NetworkMemory* dst);
};

}



