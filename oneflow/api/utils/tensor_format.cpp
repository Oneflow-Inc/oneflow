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

// Some code of this file is copied from PyTorch aten/src/ATen/core/Formatting.h

#include "oneflow/api/utils/tensor_format.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {

namespace {

// not all C++ compilers have default float so we define our own here
inline std::ios_base& defaultfloat(std::ios_base& __base) {
  __base.unsetf(std::ios_base::floatfield);
  return __base;
}

std::tuple<double, int64_t> __printFormat(std::ostream& stream, const double* self_p, size_t size) {
  if (size == 0) { return std::make_tuple(1., 0); }
  bool intMode = true;
  for (size_t i = 0; i < size; ++i) {
    auto z = self_p[i];
    if (std::isfinite(z)) {
      if (z != std::ceil(z)) {
        intMode = false;
        break;
      }
    }
  }
  int64_t offset = 0;
  while (!std::isfinite(self_p[offset])) {
    offset = offset + 1;
    if (offset == size) { break; }
  }
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  double expMin;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  double expMax;
  if (offset == size) {
    expMin = 1;
    expMax = 1;
  } else {
    expMin = fabs(self_p[offset]);
    expMax = fabs(self_p[offset]);
    for (size_t i = offset; i < size; ++i) {
      double z = fabs(self_p[i]);
      if (std::isfinite(z)) {
        if (z < expMin) { expMin = z; }
        if (self_p[i] > expMax) { expMax = z; }
      }
    }
    if (expMin != 0) {
      expMin = std::floor(std::log10(expMin)) + 1;
    } else {
      expMin = 1;
    }
    if (expMax != 0) {
      expMax = std::floor(std::log10(expMax)) + 1;
    } else {
      expMax = 1;
    }
  }
  double scale = 1;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t sz;
  if (intMode) {
    if (expMax > 9) {
      sz = 11;
      stream << std::scientific << std::setprecision(4);
    } else {
      sz = expMax + 1;
      stream << defaultfloat;
    }
  } else {
    if (expMax - expMin > 4) {
      sz = 11;
      if (std::fabs(expMax) > 99 || std::fabs(expMin) > 99) { sz = sz + 1; }
      stream << std::scientific << std::setprecision(4);
    } else {
      if (expMax > 5 || expMax < 0) {
        sz = 7;
        scale = std::pow(10, expMax - 1);
        stream << std::fixed << std::setprecision(4);
      } else {
        if (expMax == 0) {
          sz = 7;
        } else {
          sz = expMax + 6;
        }
        stream << std::fixed << std::setprecision(4);
      }
    }
  }
  return std::make_tuple(scale, sz);
}

void __printIndent(std::ostream& stream, int64_t indent) {
  for (size_t i = 0; i < indent; ++i) {
    (void)i;  // Suppress unused variable warning
    stream << " ";
  }
}

void printScale(std::ostream& stream, double scale) {
  FormatGuard guard(stream);
  stream << defaultfloat << scale << " *" << std::endl;
}

void __printMatrix(std::ostream& stream, const double* self, const Shape& shape, int64_t linesize,
                   int64_t indent) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  double scale;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t sz;
  std::tie(scale, sz) = __printFormat(stream, self, shape.Count(0));
  std::tie(scale, sz) = __printFormat(stream, self, shape.Count(0));

  __printIndent(stream, indent);
  int64_t nColumnPerLine = (linesize - indent) / (sz + 1);
  int64_t firstColumn = 0;
  int64_t lastColumn = -1;
  while (firstColumn < shape.At(1)) {
    if (firstColumn + nColumnPerLine <= shape.At(1)) {
      lastColumn = firstColumn + nColumnPerLine - 1;
    } else {
      lastColumn = shape.At(1) - 1;
    }
    if (nColumnPerLine < shape.At(1)) {
      if (firstColumn != 0) { stream << std::endl; }
      stream << "Columns " << firstColumn + 1 << " to " << lastColumn + 1 << "\n";
      __printIndent(stream, indent);
    }
    if (scale != 1) {
      printScale(stream, scale);
      __printIndent(stream, indent);
    }
    for (size_t l = 0; l < shape.At(0); ++l) {
      const double* row_ptr = self + l * (shape.Count(1));
      for (int64_t c = firstColumn; c < lastColumn + 1; c++) {
        stream << std::setw(sz) << row_ptr[c] / scale;
        if (c == lastColumn) {
          stream << std::endl;
          if (l != shape.At(0) - 1) {
            if (scale != 1) {
              __printIndent(stream, indent);
              stream << " ";
            } else {
              __printIndent(stream, indent);
            }
          }
        } else {
          stream << " ";
        }
      }
    }
    firstColumn = lastColumn + 1;
  }
}

void __printTensor(std::ostream& stream, const double* self, const Shape& shape, int64_t linesize) {
  std::vector<int64_t> counter(shape.NumAxes() - 2);
  bool start = true;
  bool finished = false;
  counter[0] = -1;
  for (size_t i = 1; i < counter.size(); ++i) { counter[i] = 0; }
  const auto& ori_dim_vec = shape.dim_vec();
  while (true) {
    for (int64_t i = 0; shape.NumAxes() - 2; i++) {
      counter[i] = counter[i] + 1;
      if (counter[i] >= shape.At(i)) {
        if (i == shape.NumAxes() - 3) {
          finished = true;
          break;
        }
        counter[i] = 0;
      } else {
        break;
      }
    }
    if (finished) { break; }
    if (start) {
      start = false;
    } else {
      stream << std::endl;
    }
    stream << "(";
    int64_t offset = 0;
    int64_t t = 0;
    for (; t < shape.NumAxes() - 2; t++) {
      offset += counter[t] * shape.Count(t + 1);
      stream << counter[t] + 1 << ",";
    }
    stream << ".,.) = " << std::endl;

    __printMatrix(stream, self + offset,
                  Shape(DimVector(ori_dim_vec.begin() + t, ori_dim_vec.end())), linesize, 1);
  }
}

}  // namespace

std::ostream& FormatTensorData(std::ostream& stream, const double* data,
                               const std::vector<int64_t>& dim_vec, int64_t linesize) {
  const auto shape = Shape(DimVector(dim_vec.begin(), dim_vec.end()));
  const auto n_elems = shape.Count(0);
  const auto n_dims = dim_vec.size();
  if (n_dims == 0) {
    stream << defaultfloat << data[0] << std::endl;
    stream << "[ "
           << "{}";
  } else if (n_dims == 1) {
    if (n_elems > 0) {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      double scale;
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      int64_t sz;
      std::tie(scale, sz) = __printFormat(stream, data, n_elems);
      if (scale != 1) { printScale(stream, scale); }
      for (auto i = 0; i < n_dims; ++i) { stream << std::setw(sz) << data[i] / scale << std::endl; }
    }
  } else if (n_dims == 2) {
    if (n_elems > 0) { __printMatrix(stream, data, shape, linesize, 0); }
  } else {
    if (n_elems > 0) { __printTensor(stream, data, shape, linesize); }
  }
  return stream;
}

}  // namespace oneflow
