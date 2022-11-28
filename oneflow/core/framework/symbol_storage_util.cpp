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
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/operator/op_node_signature.pb.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/operator/op_conf_symbol.h"
#include "oneflow/core/vm/symbol_storage.h"

namespace oneflow {

COMMAND(
    Singleton<symbol::Storage<ParallelDesc>>::SetAllocated(new symbol::Storage<ParallelDesc>()));
COMMAND(Singleton<symbol::Storage<Scope>>::SetAllocated(new symbol::Storage<Scope>()));
COMMAND(Singleton<symbol::Storage<JobDesc>>::SetAllocated(new symbol::Storage<JobDesc>()));
COMMAND(Singleton<symbol::Storage<OperatorConfSymbol>>::SetAllocated(
    new symbol::Storage<OperatorConfSymbol>()));

}  // namespace oneflow
