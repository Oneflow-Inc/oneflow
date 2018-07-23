#include <regex>
#include <string>
#include <iostream>

namespace oneflow {

bool is_ip_addr(std::string);

std::vector dns_lookup(std::string, int);

void update_job_conf(std::string);
}
