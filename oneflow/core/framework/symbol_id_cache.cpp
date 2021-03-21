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
#include "oneflow/core/framework/symbol_id_cache.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/operator/op_node_signature.cfg.h"
#include "oneflow/core/operator/op_conf.cfg.h"

namespace oneflow {

namespace symbol {

COMMAND(Global<IdCache<cfg::JobConfigProto>>::SetAllocated(new IdCache<cfg::JobConfigProto>()));
COMMAND(Global<IdCache<cfg::ParallelConf>>::SetAllocated(new IdCache<cfg::ParallelConf>()));
COMMAND(Global<IdCache<cfg::ScopeProto>>::SetAllocated(new IdCache<cfg::ScopeProto>()));
COMMAND(Global<IdCache<cfg::OpNodeSignature>>::SetAllocated(new IdCache<cfg::OpNodeSignature>()));
COMMAND(Global<IdCache<std::string>>::SetAllocated(new IdCache<std::string>()));
COMMAND(Global<IdCache<cfg::OperatorConf>>::SetAllocated(new IdCache<cfg::OperatorConf>()));

}  // namespace symbol

}  // namespace oneflow
