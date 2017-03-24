#ifndef _PATH_PATH_TYPE_H_
#define _PATH_PATH_TYPE_H_

namespace oneflow {
enum class PathType {
  kDataPath = 0,
  kModelUpdatePath,
  kModelLoadPath,
  kModelStorePath,
  kUnknownPath
};

}  // namespace oneflow
#endif  // _PATH_PATH_TYPE_H_
