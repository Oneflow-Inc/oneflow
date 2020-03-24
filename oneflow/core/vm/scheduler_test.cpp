#include <iostream>
#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/vm.h"
#include "oneflow/core/vm/stream_desc.msg.h"
#include "oneflow/core/common/object_msg_reflection.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

namespace test {

namespace {

TEST(Scheduler, __Init__) {
  auto nop_stream_desc =
      ObjectMsgPtr<StreamDesc>::New(LookupInstrTypeId("Nop").stream_type_id(), 1, 1, 1);
  auto vm_desc = ObjectMsgPtr<VmDesc>::New();
  vm_desc->mut_stream_type_id2desc()->Insert(
      ObjectMsgPtr<StreamDesc>::New(LookupInstrTypeId("NewSymbol").stream_type_id(), 1, 1, 1)
          .Mutable());
  vm_desc->mut_stream_type_id2desc()->Insert(nop_stream_desc.Mutable());
  auto scheduler = ObjectMsgPtr<Scheduler>::New(vm_desc.Get());
  ASSERT_EQ(scheduler->thread_ctx_list().size(), 2);
  ASSERT_EQ(scheduler->stream_type_id2stream_rt_desc().size(), 2);
}

TEST(Scheduler, ToDot) {
  std::string dot_str = ObjectMsgListReflection<Scheduler>().ToDot("Scheduler");
  //  std::cout << std::endl;
  //  std::cout << dot_str << std::endl;
  //  std::cout << std::endl;
}

}  // namespace

}  // namespace test

}  // namespace vm
}  // namespace oneflow
