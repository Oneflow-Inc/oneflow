/*
Copyright 2020 The OneFlow Authors. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License").
*/
#pragma once
#include "oneflow/core/device/cuda_to_hip.h"
#include <hipcub/hipcub.hpp>
// oneflow uses the cub:: namespace (cub::BlockReduce, cub::DeviceRadixSort, ...) and also
// REOPENS namespace cub to add its own reduce functors (cub::Prod, cub::Any, ...). A bare
// `namespace cub = hipcub;` alias would forbid reopening cub, so make cub a real namespace
// that pulls in hipCUB's symbols. rocThrust/hipCUB are API drop-ins.
namespace cub {
using namespace hipcub;
}
