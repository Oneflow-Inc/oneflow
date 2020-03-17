#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/object_msg_reflection.h"

namespace oneflow {
namespace vm {

namespace test {

TEST(VmDesc, ToDot) {
  std::string dot_str = ObjectMsgListReflection<VmDesc>().ToDot("VmDesc");
  //  std::cout << std::endl;
  //  std::cout << dot_str << std::endl;
  //  std::cout << std::endl;
}

}  // namespace test

}  // namespace vm
}  // namespace oneflow
