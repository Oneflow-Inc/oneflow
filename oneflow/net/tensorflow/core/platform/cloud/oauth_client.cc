/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/platform/cloud/oauth_client.h"
#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>
#include <fstream>
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/base64.h"
#include "tensorflow/core/platform/cloud/http_request.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

namespace {

// The requested lifetime of an auth bearer token.
constexpr int kRequestedTokenLifetimeSec = 3600;

// The crypto algorithm to be used with OAuth.
constexpr char kCryptoAlgorithm[] = "RS256";

// The token type for the OAuth request.
constexpr char kJwtType[] = "JWT";

// The grant type for the OAuth request. Already URL-encoded for convenience.
constexpr char kGrantType[] =
    "urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer";

Status ReadJsonValue(Json::Value json, const string& name, Json::Value* value) {
  if (!value) {
    return errors::FailedPrecondition("'value' cannot be nullptr.");
  }
  *value = json.get(name, Json::Value::null);
  if (*value == Json::Value::null) {
    return errors::FailedPrecondition(
        strings::StrCat("Couldn't read a JSON value '", name, "'."));
  }
  return Status::OK();
}

Status ReadJsonString(Json::Value json, const string& name, string* value) {
  Json::Value json_value;
  TF_RETURN_IF_ERROR(ReadJsonValue(json, name, &json_value));
  if (!json_value.isString()) {
    return errors::FailedPrecondition(
        strings::StrCat("JSON value '", name, "' is not string."));
  }
  *value = json_value.asString();
  return Status::OK();
}

Status ReadJsonInt(Json::Value json, const string& name, int64* value) {
  Json::Value json_value;
  TF_RETURN_IF_ERROR(ReadJsonValue(json, name, &json_value));
  if (!json_value.isIntegral()) {
    return errors::FailedPrecondition(
        strings::StrCat("JSON value '", name, "' is not integer."));
  }
  *value = json_value.asInt64();
  return Status::OK();
}

Status CreateSignature(RSA* private_key, StringPiece to_sign,
                       string* signature) {
  if (!private_key || !signature) {
    return errors::FailedPrecondition(
        "'private_key' and 'signature' cannot be nullptr.");
  }

  const auto md = EVP_sha256();
  if (!md) {
    return errors::Internal("Could not get a sha256 encryptor.");
  }
  std::unique_ptr<EVP_MD_CTX, std::function<void(EVP_MD_CTX*)>> md_ctx(
      EVP_MD_CTX_create(), [](EVP_MD_CTX* ptr) { EVP_MD_CTX_destroy(ptr); });
  if (!md_ctx.get()) {
    return errors::Internal("Could not create MD_CTX.");
  }

  std::unique_ptr<EVP_PKEY, std::function<void(EVP_PKEY*)>> key(
      EVP_PKEY_new(), [](EVP_PKEY* ptr) { EVP_PKEY_free(ptr); });
  EVP_PKEY_set1_RSA(key.get(), private_key);

  if (EVP_DigestSignInit(md_ctx.get(), NULL, md, NULL, key.get()) != 1) {
    return errors::Internal("DigestInit failed.");
  }
  if (EVP_DigestSignUpdate(md_ctx.get(), to_sign.data(), to_sign.size()) != 1) {
    return errors::Internal("DigestUpdate failed.");
  }
  size_t sig_len = 0;
  if (EVP_DigestSignFinal(md_ctx.get(), NULL, &sig_len) != 1) {
    return errors::Internal("DigestFinal (get signature length) failed.");
  }
  std::unique_ptr<unsigned char[]> sig(new unsigned char[sig_len]);
  if (EVP_DigestSignFinal(md_ctx.get(), sig.get(), &sig_len) != 1) {
    return errors::Internal("DigestFinal (signature compute) failed.");
  }
  EVP_MD_CTX_cleanup(md_ctx.get());
  return Base64Encode(StringPiece(reinterpret_cast<char*>(sig.get()), sig_len),
                      signature);
}

/// Encodes a claim for a JSON web token (JWT) to make an OAuth request.
Status EncodeJwtClaim(StringPiece client_email, StringPiece scope,
                      StringPiece audience, uint64 request_timestamp_sec,
                      string* encoded) {
  // Step 1: create the JSON with the claim.
  Json::Value root;
  root["iss"] = Json::Value(client_email.begin(), client_email.end());
  root["scope"] = Json::Value(scope.begin(), scope.end());
  root["aud"] = Json::Value(audience.begin(), audience.end());

  const auto expiration_timestamp_sec =
      request_timestamp_sec + kRequestedTokenLifetimeSec;

  root["iat"] = request_timestamp_sec;
  root["exp"] = expiration_timestamp_sec;

  // Step 2: represent the JSON as a string.
  string claim = root.toStyledString();

  // Step 3: encode the string as base64.
  return Base64Encode(claim, encoded);
}

/// Encodes a header for a JSON web token (JWT) to make an OAuth request.
Status EncodeJwtHeader(StringPiece key_id, string* encoded) {
  // Step 1: create the JSON with the header.
  Json::Value root;
  root["alg"] = kCryptoAlgorithm;
  root["typ"] = kJwtType;
  root["kid"] = Json::Value(key_id.begin(), key_id.end());

  // Step 2: represent the JSON as a string.
  const string header = root.toStyledString();

  // Step 3: encode the string as base64.
  return Base64Encode(header, encoded);
}

}  // namespace

OAuthClient::OAuthClient()
    : OAuthClient(
          std::unique_ptr<HttpRequest::Factory>(new HttpRequest::Factory()),
          Env::Default()) {}

OAuthClient::OAuthClient(
    std::unique_ptr<HttpRequest::Factory> http_request_factory, Env* env)
    : http_request_factory_(std::move(http_request_factory)), env_(env) {}

Status OAuthClient::GetTokenFromServiceAccountJson(
    Json::Value json, StringPiece oauth_server_uri, StringPiece scope,
    string* token, uint64* expiration_timestamp_sec) {
  if (!token || !expiration_timestamp_sec) {
    return errors::FailedPrecondition(
        "'token' and 'expiration_timestamp_sec' cannot be nullptr.");
  }
  string private_key_serialized, private_key_id, client_id, client_email;
  TF_RETURN_IF_ERROR(
      ReadJsonString(json, "private_key", &private_key_serialized));
  TF_RETURN_IF_ERROR(ReadJsonString(json, "private_key_id", &private_key_id));
  TF_RETURN_IF_ERROR(ReadJsonString(json, "client_id", &client_id));
  TF_RETURN_IF_ERROR(ReadJsonString(json, "client_email", &client_email));

  std::unique_ptr<BIO, std::function<void(BIO*)>> bio(
      BIO_new(BIO_s_mem()), [](BIO* ptr) { BIO_free_all(ptr); });
  if (BIO_puts(bio.get(), private_key_serialized.c_str()) !=
      static_cast<int>(private_key_serialized.size())) {
    return errors::Internal("Could not load the private key.");
  }
  std::unique_ptr<RSA, std::function<void(RSA*)>> private_key(
      PEM_read_bio_RSAPrivateKey(bio.get(), nullptr, nullptr, nullptr),
      [](RSA* ptr) { RSA_free(ptr); });
  if (!private_key.get()) {
    return errors::Internal("Could not deserialize the private key.");
  }

  const uint64 request_timestamp_sec = env_->NowSeconds();

  string encoded_claim, encoded_header;
  TF_RETURN_IF_ERROR(EncodeJwtHeader(private_key_id, &encoded_header));
  TF_RETURN_IF_ERROR(EncodeJwtClaim(client_email, scope, oauth_server_uri,
                                    request_timestamp_sec, &encoded_claim));
  const string to_sign = encoded_header + "." + encoded_claim;
  string signature;
  TF_RETURN_IF_ERROR(CreateSignature(private_key.get(), to_sign, &signature));
  const string jwt = to_sign + "." + signature;
  const string request_body =
      strings::StrCat("grant_type=", kGrantType, "&assertion=", jwt);

  // Send the request to the Google OAuth 2.0 server to get the token.
  std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
  std::unique_ptr<char[]> response_buffer(new char[kResponseBufferSize]);
  StringPiece response;
  TF_RETURN_IF_ERROR(request->Init());
  TF_RETURN_IF_ERROR(request->SetUri(oauth_server_uri.ToString()));
  TF_RETURN_IF_ERROR(
      request->SetPostRequest(request_body.c_str(), request_body.size()));
  TF_RETURN_IF_ERROR(request->SetResultBuffer(response_buffer.get(),
                                              kResponseBufferSize, &response));
  TF_RETURN_IF_ERROR(request->Send());

  TF_RETURN_IF_ERROR(ParseOAuthResponse(response, request_timestamp_sec, token,
                                        expiration_timestamp_sec));
  return Status::OK();
}

Status OAuthClient::GetTokenFromRefreshTokenJson(
    Json::Value json, StringPiece oauth_server_uri, string* token,
    uint64* expiration_timestamp_sec) {
  if (!token || !expiration_timestamp_sec) {
    return errors::FailedPrecondition(
        "'token' and 'expiration_timestamp_sec' cannot be nullptr.");
  }
  string client_id, client_secret, refresh_token;
  TF_RETURN_IF_ERROR(ReadJsonString(json, "client_id", &client_id));
  TF_RETURN_IF_ERROR(ReadJsonString(json, "client_secret", &client_secret));
  TF_RETURN_IF_ERROR(ReadJsonString(json, "refresh_token", &refresh_token));

  const auto request_body = strings::StrCat(
      "client_id=", client_id, "&client_secret=", client_secret,
      "&refresh_token=", refresh_token, "&grant_type=refresh_token");

  const uint64 request_timestamp_sec = env_->NowSeconds();

  std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
  std::unique_ptr<char[]> response_buffer(new char[kResponseBufferSize]);
  StringPiece response;
  TF_RETURN_IF_ERROR(request->Init());
  TF_RETURN_IF_ERROR(request->SetUri(oauth_server_uri.ToString()));
  TF_RETURN_IF_ERROR(
      request->SetPostRequest(request_body.c_str(), request_body.size()));
  TF_RETURN_IF_ERROR(request->SetResultBuffer(response_buffer.get(),
                                              kResponseBufferSize, &response));
  TF_RETURN_IF_ERROR(request->Send());

  TF_RETURN_IF_ERROR(ParseOAuthResponse(response, request_timestamp_sec, token,
                                        expiration_timestamp_sec));
  return Status::OK();
}

Status OAuthClient::ParseOAuthResponse(StringPiece response,
                                       uint64 request_timestamp_sec,
                                       string* token,
                                       uint64* expiration_timestamp_sec) {
  if (!token || !expiration_timestamp_sec) {
    return errors::FailedPrecondition(
        "'token' and 'expiration_timestamp_sec' cannot be nullptr.");
  }
  Json::Value root;
  Json::Reader reader;
  if (!reader.parse(response.begin(), response.end(), root)) {
    return errors::Internal("Couldn't parse JSON response from OAuth server.");
  }

  string token_type;
  TF_RETURN_IF_ERROR(ReadJsonString(root, "token_type", &token_type));
  if (token_type != "Bearer") {
    return errors::FailedPrecondition("Unexpected Oauth token type: " +
                                      token_type);
  }
  int64 expires_in;
  TF_RETURN_IF_ERROR(ReadJsonInt(root, "expires_in", &expires_in));
  *expiration_timestamp_sec = request_timestamp_sec + expires_in;
  TF_RETURN_IF_ERROR(ReadJsonString(root, "access_token", token));

  return Status::OK();
}

}  // namespace tensorflow
