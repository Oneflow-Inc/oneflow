include(FetchContent)

set_mirror_url_with_hash(
  string_view_URL "https://github.com/martinmoene/string-view-lite/archive/refs/tags/v1.6.0.tar.gz"
  2bdbfa2def2d460e3f78ee9846ece943)

FetchContent_Declare(string_view URL ${string_view_URL} URL_HASH MD5=${string_view_URL_HASH})

FetchContent_MakeAvailable(string_view)
