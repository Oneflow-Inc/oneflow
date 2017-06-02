#include "register/register_manager.h"
#include "register/blob.h"
#include "common/id_manager.h"

namespace oneflow {

void RegstMgr::NewRegsts(const RegstDescProto& regst_desc_proto,
               std::function<void(Regst*)> OneRegstDone) {
  std::shared_ptr<RtRegstDesc> runtime_regst_desc(new RtRegstDesc(regst_desc_proto));
  for (size_t i = 0; i < regst_desc_proto.register_num(); ++i) {
    Regst* regst = new Regst;
    regst->regst_desc_ = runtime_regst_desc;
    regst->regst_id_ = IDMgr::Singleton().NewRegstId(regst_desc_proto.regst_desc_id());
    std::vector<std::function<void()>> deallocates;
    deallocates.reserve(regst_desc_proto.lbn2shape().size());
    for (const auto& pair : regst_desc_proto.lbn2shape()) {
      Shape* shape_ptr = runtime_regst_desc->GetShapePtrFromLbn(pair.first);
      std::pair<char*, std::function<void()>> allocation = 
          MemoryAllocator::Singleton().Allocate(regst_desc_proto.mem_case(), shape_ptr->elem_cnt());
      deallocates.push_back(allocation.second);
      CHECK(regst->lbn2blob_.emplace(pair.first, of_make_unique<Blob>(allocation.first, shape_ptr)).second);
    }
    regst->deleter_ = [&deallocates]() {
      for (std::function<void()> func : deallocates) {
        func();
      }
    };
    OneRegstDone(regst);
  }
}

}
