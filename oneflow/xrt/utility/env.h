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
#ifndef ONEFLOW_XRT_UTILITY_ENV_H_
#define ONEFLOW_XRT_UTILITY_ENV_H_

#include <stdlib.h>
#include <string.h>
#include <string>

// Refer to glog `src/base/commandlineflags.h`

#define EnvToString(envname, dflt) (!getenv(#envname) ? (dflt) : getenv(#envname))

#define EnvToBool(envname, dflt) \
  (!getenv(#envname) ? (dflt) : memchr("tTyY1\0", getenv(#envname)[0], 6) != NULL)

#define EnvToInt(envname, dflt) (!getenv(#envname) ? (dflt) : strtol(getenv(#envname), NULL, 10))

#define EnvToInt64(envname, dflt) (!getenv(#envname) ? (dflt) : strtoll(getenv(#envname), NULL, 10))

#endif  // ONEFLOW_XRT_UTILITY_ENV_H_
