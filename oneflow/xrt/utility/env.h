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
