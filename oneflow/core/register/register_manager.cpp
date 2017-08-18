#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

std::vector<NetMemoryDescriptor> RegstMgr::NewRegsts(
    const RegstDescProto& regst_desc_proto,
    std::function<void(Regst*)> OneRegstDone) {
  std::vector<NetMemoryDescriptor> net_memory_descs;
  const RtRegstDesc* runtime_regst_desc = new RtRegstDesc(regst_desc_proto);
  rt_regst_descs_.emplace_back(runtime_regst_desc);
  for (int64_t i = 0; i < regst_desc_proto.register_num(); ++i) {
    Regst* regst = new Regst;
    regst->regst_desc_ = runtime_regst_desc;

    size_t elem_size = sizeof(float);
    if (JobDesc::Singleton()->floating_point_type() == kDouble) {
      elem_size = sizeof(double);
    }
    int64_t elem_cnt = 0;
    std::vector<std::string> lbns;
    lbns.reserve(regst_desc_proto.lbn2shape().size());
    for (const auto& pair : regst_desc_proto.lbn2shape()) {
      const Shape* shape_ptr =
          runtime_regst_desc->GetShapePtrFromLbn(pair.first);
      lbns.push_back(pair.first);
      elem_cnt += shape_ptr->elem_cnt();
    }
    std::sort(lbns.begin(), lbns.end());
    std::tuple<char*, std::function<void()>, void*> allocation =
        MemoryAllocator::Singleton()->Allocate(regst_desc_proto.mem_case(),
                                               elem_cnt * elem_size);

    int64_t blob_idx = 0;
    for (const std::string& lbn : lbns) {
      const Shape* shape_ptr = runtime_regst_desc->GetShapePtrFromLbn(lbn);
      auto blob_ptr =
          of_make_unique<Blob>(std::get<0>(allocation) + blob_idx, shape_ptr);
      CHECK(regst->lbn2blob_.emplace(lbn, std::move(blob_ptr)).second);
      blob_idx += shape_ptr->elem_cnt() * elem_size;
    }
    Shape* packed_blob_shape = new Shape({elem_cnt});
    regst->packed_blob_.reset(
        new Blob(std::get<0>(allocation), packed_blob_shape));
    regst->deleter_ = [allocation, packed_blob_shape]() {
      std::get<1>(allocation)();
      delete packed_blob_shape;
    };
    if (std::get<2>(allocation) != nullptr) {
      // Leave this_machine_id and the consumer_machine_ids as uninitialized
      NetMemoryDescriptor net_memory_desc{
          regst, std::get<0>(allocation), std::get<2>(allocation), -1, {}};
      net_memory_descs.push_back(net_memory_desc);
    }
    OneRegstDone(regst);
  }
  return net_memory_descs;
}

}  // namespace oneflow
