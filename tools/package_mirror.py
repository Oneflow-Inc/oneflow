import glob
import argparse
import os
import re
from urllib.parse import urlparse
import requests
import hashlib
import base64
import tempfile

try:
    import oss2
except:
    pass


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--src_path", type=str, required=False)
parser.add_argument("-u", "--url", type=str, required=False)
args = parser.parse_args()


def glob_by_pattern(dir_path, pattern):
    result = []
    for x in glob.glob(os.path.join(dir_path, pattern), recursive=True):
        result.append(x)
    return result


def scan_urls(dir_path):
    cmakes = glob_by_pattern(dir_path, "**/*.cmake")
    cmakes += glob_by_pattern(dir_path, "**/*.bzl")
    urls = []
    for cmake_path in cmakes:
        with open(cmake_path) as f:
            content = f.read()
            urls += re.findall(r'https?://[^\s<>"\)]+|www\.[^\s<>"]+', content)
    return urls


def convert_url_to_oss_key(url):
    parsed = urlparse(url)
    assert parsed.scheme == "https", url
    assert not parsed.params
    assert not parsed.query
    assert not parsed.port
    assert not parsed.fragment
    assert parsed.path.startswith("/")
    path = parsed.path[1::]
    ret = os.path.join("third_party_mirror", parsed.scheme, parsed.netloc, path)
    assert convert_url_to_oss_key1(url) == ret
    return ret


def convert_url_to_oss_key1(url: str):
    https = "https://"
    assert url.startswith(https)
    path = url[len(https) :]
    return "/".join(["third_party_mirror", "https", path])


def convert_url_to_oss_https_url(url):
    if should_be_mirrored(url):
        key = convert_url_to_oss_key(url)
        prefix = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/"
        return os.path.join(prefix, key)
    else:
        return url


def should_be_mirrored(url: str):
    parsed = urlparse(url)
    return (
        not parsed.port
        and not parsed.query
        and not parsed.params
        and url.endswith(("gz", "tar", "zip"))
        and not "mirror.tensorflow.org" in url
        and not "mirror.bazel.build" in url
        and not "aliyuncs.com" in url
    )


def calculate_data_md5(data):
    md5 = hashlib.md5()
    md5.update(data)
    digest = md5.digest()
    return base64.b64encode(digest)


def upload_one_to_aliyun(url: str):
    ki = os.getenv("OSS_ACCESS_KEY_ID")
    ks = os.getenv("OSS_ACCESS_KEY_SECRET")
    auth = oss2.Auth(ki, ks)
    endpoint = "oss-cn-beijing.aliyuncs.com"
    bucket = oss2.Bucket(auth, endpoint, "oneflow-static")
    key = convert_url_to_oss_key(url)

    if bucket.object_exists(key):
        print("exists: ", key)
    else:
        content = requests.get(url).content
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with tempfile.NamedTemporaryFile() as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                headers = {}
                bucket.put_object_from_file(key, f.name, headers=headers)


def upload_to_aliyun(dir_path):
    urls = scan_urls(dir_path)
    for url in urls:
        if should_be_mirrored(url):
            print("mirroring: ", url)
            upload_one_to_aliyun(url)
        else:
            print("skipped: ", url)
            continue


if __name__ == "__main__":
    if args.src_path != None:
        upload_to_aliyun(args.src_path)
    if args.url != None:
        oss_url = convert_url_to_oss_https_url(args.url)
        print(oss_url)
