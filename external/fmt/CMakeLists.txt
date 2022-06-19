include(FetchContent)
FetchContent_Declare(
  fmt
  URL ${FMT_URL}
  URL_HASH MD5=${FMT_MD5}
)

FetchContent_MakeAvailable(fmt)

set_property(GLOBAL PROPERTY FMT_SOURCE_DIR "${fmt_SOURCE_DIR}")
