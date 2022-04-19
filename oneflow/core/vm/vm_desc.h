/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_VM_MEM_ZONE_TYPE_DESC__H_
#define ONEFLOW_CORE_VM_MEM_ZONE_TYPE_DESC__H_

#include "oneflow/core/vm/stream_desc.h"
#include "oneflow/core/vm/virtual_machine_engine.h"
#include "oneflow/core/vm/vm_resource_desc.h"
#include "oneflow/core/common/range.h"

namespace oneflow {
namespace vm {

class VmDesc final : public intrusive::Base {
 public:
  // types
  using StreamType2StreamDesc = intrusive::SkipList<INTRUSIVE_FIELD(StreamDesc, stream_type_key_)>;
  // Getters
  const VmResourceDesc& vm_resource_desc() const {
    if (vm_resource_desc_) { return vm_resource_desc_.Get(); }
    static const auto default_val = intrusive::make_shared<VmResourceDesc>();
    return default_val.Get();
  }
  const Range& machine_id_range() const { return machine_id_range_; }
  const StreamType2StreamDesc& stream_type2desc() const { return stream_type2desc_; }
  // Setters
  VmResourceDesc* mut_vm_resource_desc() {
    if (!vm_resource_desc_) { vm_resource_desc_ = intrusive::make_shared<VmResourceDesc>(); }
    return vm_resource_desc_.Mutable();
  }
  Range* mut_machine_id_range() { return &machine_id_range_; }
  StreamType2StreamDesc* mut_stream_type2desc() { return &stream_type2desc_; }

  // methods
  void __Init__(const VmResourceDesc& vm_resource_desc) { __Init__(vm_resource_desc, Range(0, 1)); }
  void __Init__(const VmResourceDesc& vm_resource_desc, const Range& machine_id_range) {
    mut_vm_resource_desc()->CopyFrom(vm_resource_desc);
    *mut_machine_id_range() = machine_id_range;
  }

 private:
  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  VmDesc() : intrusive_ref_(), vm_resource_desc_(), machine_id_range_(), stream_type2desc_() {}
  intrusive::Ref intrusive_ref_;
  // fields
  intrusive::shared_ptr<VmResourceDesc> vm_resource_desc_;
  Range machine_id_range_;
  // maps
  StreamType2StreamDesc stream_type2desc_;
};

intrusive::shared_ptr<VmDesc> MakeVmDesc(const Resource& resource, int64_t this_machine_id);
intrusive::shared_ptr<VmDesc> MakeVmDesc(const Resource& resource, int64_t this_machine_id,
                                         const std::set<std::string>& instr_type_names);

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_MEM_ZONE_TYPE_DESC__H_
