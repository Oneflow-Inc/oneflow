include(FetchContent)

set_mirror_url_with_hash(
  string_view_URL
  "https://github.com/martinmoene/string-view-lite/archive/3d2a4a7ebcc5dbdd55bd00026075c574a13c861b.tar.gz"
  ff90f618263ec4abbe4ae89eb0d1660d)

FetchContent_Declare(string_view URL ${string_view_URL} URL_HASH MD5=${string_view_URL_HASH})

FetchContent_MakeAvailable(string_view)
