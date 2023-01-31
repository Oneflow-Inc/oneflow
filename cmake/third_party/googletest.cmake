include(FetchContent)

set_mirror_url_with_hash(
  googletest_URL
  https://github.com/google/googletest/archive/e2239ee6043f73722e7aa812a459f54a28552929.tar.gz
  7ad24b4f2a0e895f0a11f25cefd39b1e)

FetchContent_Declare(googletest URL ${googletest_URL} URL_HASH MD5=${googletest_URL_HASH})

FetchContent_MakeAvailable(googletest)
