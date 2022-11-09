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
#ifndef ONEFLOW_CORE_COMMON_PLATFORM_H_
#define ONEFLOW_CORE_COMMON_PLATFORM_H_

// Set one OF_PLATFORM_* macro and set OF_IS_MOBILE_PLATFORM if the platform is for
// mobile.

#if !defined(OF_PLATFORM_POSIX) && !defined(OF_PLATFORM_GOOGLE)                    \
    && !defined(OF_PLATFORM_POSIX_ANDROID) && !defined(OF_PLATFORM_GOOGLE_ANDROID) \
    && !defined(OF_PLATFORM_WINDOWS)

// Choose which platform we are on.
#if defined(ANDROID) || defined(__ANDROID__)
#define OF_PLATFORM_POSIX_ANDROID
#define OF_IS_MOBILE_PLATFORM

#elif defined(__APPLE__)
#define OF_PLATFORM_POSIX
#include "TargetConditionals.h"
#if OF_TARGET_IPHONE_SIMULATOR
#define OF_IS_MOBILE_PLATFORM
#elif OF_TARGET_OS_IPHONE
#define OF_IS_MOBILE_PLATFORM
#endif

#elif defined(_WIN32)
#define OF_PLATFORM_WINDOWS

#elif defined(__arm__)
#define OF_PLATFORM_POSIX

// Require an outside macro to tell us if we're building for Raspberry Pi.
#if !defined(RASPBERRY_PI)
#define OF_IS_MOBILE_PLATFORM
#endif  // !defined(RASPBERRY_PI)

#else
// If no platform specified, use:
#define OF_PLATFORM_POSIX

#endif
#endif

// Look for both gcc/clang and Visual Studio macros indicating we're compiling
// for an x86 device.
#if defined(__x86_64__) || defined(__amd64__) || defined(_M_IX86) || defined(_M_X64)
#define OF_PLATFORM_IS_X86
#endif

#endif  // ONEFLOW_CORE_COMMON_PLATFORM_H_
