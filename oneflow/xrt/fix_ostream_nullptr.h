#include <iostream>

namespace std {
    ostream& operator<<(ostream& os, nullptr_t) {
        return os << "nullptr";
    }
}
