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
#ifndef ONEFLOW_CORE_EMBEDDING_HASH_FUNCTION_H_
#define ONEFLOW_CORE_EMBEDDING_HASH_FUNCTION_H_

#include <stdint.h>
#include "oneflow/core/common/data_type.h"

namespace oneflow {

namespace embedding {

namespace {

// From https://github.com/Cyan4973/xxHash/blob/dev/xxhash.h
static const uint64_t PRIME64_1 =
    0x9E3779B185EBCA87ULL;  // 0b1001111000110111011110011011000110000101111010111100101010000111
static const uint64_t PRIME64_2 =
    0xC2B2AE3D27D4EB4FULL;  // 0b1100001010110010101011100011110100100111110101001110101101001111
static const uint64_t PRIME64_3 =
    0x165667B19E3779F9ULL;  // 0b0001011001010110011001111011000110011110001101110111100111111001
static const uint64_t PRIME64_4 =
    0x85EBCA77C2B2AE63ULL;  // 0b1000010111101011110010100111011111000010101100101010111001100011
static const uint64_t PRIME64_5 =
    0x27D4EB2F165667C5ULL;  // 0b0010011111010100111010110010111100010110010101100110011111000101

#define XXH_rotl64(x, r) (((x) << (r)) | ((x) >> (64 - (r))))

OF_DEVICE_FUNC uint64_t XXH64_round(uint64_t acc, uint64_t input) {
  acc += input * PRIME64_2;
  acc = XXH_rotl64(acc, 31);
  acc *= PRIME64_1;
  return acc;
}

OF_DEVICE_FUNC uint64_t xxh64_uint64(uint64_t v, uint64_t seed) {
  uint64_t acc = seed + PRIME64_5;
  acc += sizeof(uint64_t);
  acc = acc ^ XXH64_round(0, v);
  acc = XXH_rotl64(acc, 27) * PRIME64_1;
  acc = acc + PRIME64_4;
  acc ^= (acc >> 33);
  acc = acc * PRIME64_2;
  acc = acc ^ (acc >> 29);
  acc = acc * PRIME64_3;
  acc = acc ^ (acc >> 32);
  return acc;
}

static const size_t kShardingHashSeed = 1;
static const size_t kLocalUniqueHashSeed = 2;
static const size_t kGlobalUniqueHashSeed = 3;
static const size_t kFullCacheHashSeed = 4;
static const size_t kLruCacheHashSeed = 5;

}  // namespace

struct ShardingHash {
  OF_DEVICE_FUNC size_t operator()(uint64_t v) { return xxh64_uint64(v, kShardingHashSeed); }
  OF_DEVICE_FUNC size_t operator()(uint32_t v) { return xxh64_uint64(v, kShardingHashSeed); }
  OF_DEVICE_FUNC size_t operator()(int32_t v) {
    return xxh64_uint64(static_cast<uint32_t>(v), kShardingHashSeed);
  }
  OF_DEVICE_FUNC size_t operator()(int64_t v) {
    return xxh64_uint64(static_cast<uint64_t>(v), kShardingHashSeed);
  }
};

struct LocalUniqueHash {
  OF_DEVICE_FUNC size_t operator()(uint64_t v) { return xxh64_uint64(v, kLocalUniqueHashSeed); }
};

struct GlobalUniqueHash {
  OF_DEVICE_FUNC size_t operator()(uint64_t v) { return xxh64_uint64(v, kGlobalUniqueHashSeed); }
};

struct FullCacheHash {
  OF_DEVICE_FUNC size_t operator()(uint64_t v) { return xxh64_uint64(v, kFullCacheHashSeed); }
};

struct LruCacheHash {
  OF_DEVICE_FUNC size_t operator()(uint64_t v) { return xxh64_uint64(v, kLruCacheHashSeed); }
};

}  // namespace embedding
}  // namespace oneflow
#endif  // ONEFLOW_CORE_EMBEDDING_HASH_FUNCTION_H_
