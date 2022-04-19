include(FetchContent)

set_mirror_url_with_hash(
  googletest_URL https://github.com/google/googletest/archive/release-1.11.0.tar.gz
  e8a8df240b6938bb6384155d4c37d937)

FetchContent_Declare(googletest URL ${googletest_URL} URL_HASH MD5=${googletest_URL_HASH})

FetchContent_MakeAvailable(googletest)
