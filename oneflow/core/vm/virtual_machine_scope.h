#include "oneflow/core/job/resource.pb.h"

namespace oneflow {
namespace vm {

class VirtualMachineScope {
 public:
  VirtualMachineScope(const Resource& resource);
  ~VirtualMachineScope();
};

}
}
