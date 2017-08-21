#ifndef ONEFLOW_CORE_PERSISTENCE_PLATFORM_H_
#define ONEFLOW_CORE_PERSISTENCE_PLATFORM_H_

// Set one PLATFORM_* macro and set IS_MOBILE_PLATFORM if the platform is for
// mobile.

#if !defined(PLATFORM_POSIX) && !defined(PLATFORM_GOOGLE)                    \
    && !defined(PLATFORM_POSIX_ANDROID) && !defined(PLATFORM_GOOGLE_ANDROID) \
    && !defined(PLATFORM_WINDOWS)

// Choose which platform we are on.
#if defined(ANDROID) || defined(__ANDROID__)
#define PLATFORM_POSIX_ANDROID
#define IS_MOBILE_PLATFORM

#elif defined(__APPLE__)
#define PLATFORM_POSIX
#include "TargetConditionals.h"
#if TARGET_IPHONE_SIMULATOR
#define IS_MOBILE_PLATFORM
#elif TARGET_OS_IPHONE
#define IS_MOBILE_PLATFORM
#endif

#elif defined(_WIN32)
#define PLATFORM_WINDOWS

#elif defined(__arm__)
#define PLATFORM_POSIX

// Require an outside macro to tell us if we're building for Raspberry Pi.
#if !defined(RASPBERRY_PI)
#define IS_MOBILE_PLATFORM
#endif  // !defined(RASPBERRY_PI)

#else
// If no platform specified, use:
#define PLATFORM_POSIX

#endif
#endif

// Look for both gcc/clang and Visual Studio macros indicating we're compiling
// for an x86 device.
#if defined(__x86_64__) || defined(__amd64__) || defined(_M_IX86) \
    || defined(_M_X64)
#define PLATFORM_IS_X86
#endif

#endif  // ONEFLOW_CORE_PERSISTENCE_PLATFORM_H_
