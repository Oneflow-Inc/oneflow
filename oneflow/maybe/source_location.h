#ifndef ONEFLOW_MAYBE_SOURCE_LOCATION_H_
#define ONEFLOW_MAYBE_SOURCE_LOCATION_H_

#include <cstdint>

namespace oneflow {

class source_location {
 public:
#if not defined(__apple_build_version__) and defined(__clang__) and (__clang_major__ >= 9)
  static constexpr source_location current(const char* file = __builtin_FILE(),
                                           const char* func = __builtin_FUNCTION(),
                                           std::uint_least32_t line = __builtin_LINE(),
                                           std::uint_least32_t col = 0) noexcept
#elif defined(__GNUC__) and (__GNUC__ > 4 or (__GNUC__ == 4 and __GNUC_MINOR__ >= 8))
  static constexpr source_location current(const char* file = __builtin_FILE(),
                                           const char* func = __builtin_FUNCTION(),
                                           std::uint_least32_t line = __builtin_LINE(),
                                           std::uint_least32_t col = 0) noexcept
#else
  static constexpr source_location current(const char* file = "unsupported",
                                           const char* func = "unsupported",
                                           std::uint_least32_t line = 0,
                                           std::uint_least32_t col = 0) noexcept
#endif
  {
    return source_location(file, func, line, col);
  }

  constexpr const char* file_name() const noexcept { return file_; }
  constexpr const char* function_name() const noexcept { return func_; }
  constexpr uint_least32_t line() const noexcept { return line_; }
  constexpr uint_least32_t column() const noexcept { return col_; }

 private:
  constexpr source_location(const char* file, const char* func, std::uint_least32_t line,
                            std::uint_least32_t col)
      : file_(file), func_(func), line_(line), col_(col) {}

  const char* file_;
  const char* func_;
  const std::uint_least32_t line_;
  const std::uint_least32_t col_;
};

}  // namespace oneflow

#endif  // ONEFLOW_MAYBE_SOURCE_LOCATION_H_
