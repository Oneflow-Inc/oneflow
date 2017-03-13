#ifndef _PATH_PATH_TYPE_H_
#define _PATH_PATH_TYPE_H_

namespace caffe {
enum class PathType {
  kDataPath = 0,
  kModelUpdatePath,
  kModelLoadPath,
  kModelStorePath,
  kUnknownPath
};

}  // namespace caffe
#endif  // _PATH_PATH_TYPE_H_
