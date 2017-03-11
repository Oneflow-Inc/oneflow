/*
* this tool is used to convert images to a binary record,
* which includes image datas and labels
*/

#include <sstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include "io/io.h"

// count file line number
uint32_t countline(std::string path) {
  std::ifstream fi(path);
  CHECK(fi.fail() == 0) << "failed to open:" << path;
  uint32_t count = 0;
  char buf[1024];
  while (fi.getline(buf, 1024)) {
    // skip empty line
    if (fi.gcount() < 3) continue;
    count++;
  }
  fi.close();
  return count;
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    printf("Usage: <image.lst> <image_root_dir> <output_name>"\
      "[additional parameters in form key=value]\n"\
      "Possible additional parameters:\n"\
      "\tnsplit=NSPLIT[default=1] used for part generation, "\
      "logically split the image.list to NSPLIT parts by position\n"\
      "\tpart=PART[default=0] used for part generation, "\
      "pack the images from the specific part in image.list\n"\
      "\tshuffle=SHUFFLE[default=0] randomly shuffle the data, "\
      "0 for false and 1 for true\n");
    return 0;
  }

  int nsplit = 1;
  int partid = 0;
  int do_shuffle = 0;
  for (int i = 4; i < argc; ++i) {
    char key[128], val[128];
    if (sscanf(argv[i], "%[^=]=%s", key, val) == 2) {
      if (!strcmp(key, "nsplit")) nsplit = atoi(val);
      if (!strcmp(key, "part")) partid = atoi(val);
      if (!strcmp(key, "shuffle")) do_shuffle = atoi(val);
    }
  }
  // todo: split the image.list

  using namespace caffe;
  std::string flistname = argv[1];
  std::string root = argv[2];
  std::string outname = argv[3];
  const uint32_t image_num = countline(flistname);  // images total number

  std::ostringstream os;
  if (nsplit == 1) {
    os << outname << ".bin";
  } else {
    os << outname << ".part" << std::setw(3) << std::setfill('0')
      << partid << ".bin";
  }
  LOG(INFO) << "Write to output: " << os.str();

  BinaryOutputStream fo;
  fo.Open(os.str());

  std::string fname, path;  // image path
  int32_t image_idx, image_label;  // image id and label

  std::ifstream flist(flistname);
  CHECK(flist.fail() == 0) << "open image list failed: " << flistname;
  std::vector<std::string> list_vector;
  list_vector.reserve(image_num);
  std::string fline;
  LOG(INFO) << "Reading list from: " << flistname;
  while (std::getline(flist, fline)) {
    // skip empty line
    if (fline.size() < 3) continue;
    list_vector.push_back(fline);
  }
  LOG(INFO) << "list already read ";
  flist.close();
  CHECK_EQ(list_vector.size(), image_num);

  std::ofstream out_list_shuffed;
  if (do_shuffle > 0) {
    LOG(INFO) << "shuffle list";
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    shuffle(list_vector.begin(), list_vector.end(),
      std::default_random_engine(seed));
    // int pos = flistname.find(".");
    // std::string shuffed_name = flistname.substr(0, pos) + "_shuffed.txt";
    std::string shuffed_name = outname + "_shuffed.txt";
    out_list_shuffed.open(shuffed_name);
    CHECK(out_list_shuffed.good());
  }

  uint64_t *offsetbuf = new uint64_t[image_num + 1];
  offsetbuf[0] = 0;
  std::vector<char> decode_buf;

  fo.WriteEmptyHeader(offsetbuf, image_num);

  std::string line;
  uint32_t idx = 0;
  std::vector<std::string>::iterator list_iter = list_vector.begin();

  while (list_iter != list_vector.end() && (image_num > idx)) {
    line = *list_iter;
    if (do_shuffle > 0) {
      out_list_shuffed << line << std::endl;
    }
    std::istringstream is(line);
    if (!(is >> image_idx >> image_label)) continue;
    CHECK(std::getline(is, fname)) << "get imagename error at line: " << idx;
    // eliminate invalid chars in the end
    while (fname.length() != 0 &&
      (isspace(*fname.rbegin()) || !isprint(*fname.rbegin()))) {
      fname.resize(fname.length() - 1);
    }
    // eliminate invalid chars in beginning.
    const char *p = fname.c_str();
    while (isspace(*p)) ++p;
    path = root + p;

    std::ifstream fi(path, std::ifstream::binary);
    CHECK(fi.fail() == 0) << "open image failed: " << path;

    // get file length
    fi.seekg(0, std::ios_base::end);
    uint64_t length = fi.tellg();
    fi.seekg(0, std::ios_base::beg);

    decode_buf.clear();
    decode_buf.resize(length);
    fi.read(&decode_buf[0], length);
    fi.close();

    fo.WriteBlob(image_label, &decode_buf[0], length);
    offsetbuf[idx + 1] = offsetbuf[idx] + length + sizeof(int32_t);

    idx++;
    list_iter++;
    if (idx % 1000 == 0) {
      LOG(INFO) << idx << " images processed";
    }
  }
  fo.WriteRealHeader(offsetbuf, image_num);

  fo.Close();
  if (do_shuffle > 0) {
    out_list_shuffed.close();
  }

  LOG(INFO) << "all images processed!";

  return 0;
}

