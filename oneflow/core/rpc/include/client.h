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

#ifndef ONEFLOW_CORE_RPC_INCLUDE_CLIENT_
#define ONEFLOW_CORE_RPC_INCLUDE_CLIENT_

#ifdef RPC_CLIENT_GRPC
#include "oneflow/core/rpc/include/gprc/rpc_client.h"
#endif  // RPC_CLIENT_GRPC

#ifdef RPC_CLIENT_LOCAL
#include "oneflow/core/rpc/include/local/rpc_client.h"
#endif  // RPC_CLIENT_LOCAL

#endif  // ONEFLOW_CORE_RPC_INCLUDE_CLIENT_
