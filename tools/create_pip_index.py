# python3 -m pip install oss2 beautifulsoup4 --user.
from bs4 import BeautifulSoup
import os
import oss2


page_template = """
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">
<html>

<head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <title>Directory listing for /oneflow/</title>
</head>

<body>
    <h1>Directory listing for /oneflow/</h1>
    <hr>
    <ul>
    </ul>
    <hr>
</body>

</html>
"""
soup = BeautifulSoup(page_template, "html.parser")


def url4key(endpoint, bucket, key):
    return "https://{}.{}/{}".format(bucket, endpoint, key)


def append_link(soup, link):
    li_tag = soup.new_tag("li")
    soup.body.ul.append(li_tag)

    a_tag = soup.new_tag("a", href=link)
    a_tag.append(os.path.basename(link))
    li_tag.append(a_tag)


def generate_index_file(endpoint, bucket, dir_key, file_path, index_key=None):
    ki = os.getenv("OSS_ACCESS_KEY_ID")
    ks = os.getenv("OSS_ACCESS_KEY_SECRET")
    auth = oss2.Auth(ki, ks)
    bucket_obj = oss2.Bucket(auth, endpoint, bucket)

    files = bucket_obj.list_objects(dir_key + "/")
    count = 0
    for f in files.object_list:
        key = f.key
        print(key)
        link = url4key(endpoint, bucket, key)
        append_link(soup, link)
        count += 1
    assert count
    html = soup.prettify()
    with open(file_path, "w+") as f:
        f.write(html)
    if index_key == None:
        index_key = dir_key + ".pip.index.html"
    bucket_obj.put_object_from_file(index_key, file_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output_path", type=str, required=False, default="pip_index.html"
    )
    parser.add_argument(
        "-e",
        "--endpoint",
        type=str,
        required=False,
        default="oss-cn-beijing.aliyuncs.com",
    )
    parser.add_argument(
        "-b", "--bucket", type=str, required=False, default="oneflow-public",
    )
    parser.add_argument(
        "-d", "--dir_key", type=str, required=False, default="nightly",
    )
    parser.add_argument("--index_key", type=str, required=False, default=None)
    args = parser.parse_args()
    assert args.dir_key[-1] != "/"
    generate_index_file(
        args.endpoint,
        args.bucket,
        args.dir_key,
        args.output_path,
        index_key=args.index_key,
    )
