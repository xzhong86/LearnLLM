
#include "llm-utils.hpp"

#include <fcntl.h>     // for O_RDONLY
#include <unistd.h>    // for poxis open
#include <sys/mman.h>  // for poxis mmap

namespace llm {

OSMemMap::~OSMemMap() {
    if (fd_ != -1) {
        if (ptr_ != MAP_FAILED) { munmap(ptr_, file_size_); }
        close(fd_);
    }
}

void OSMemMap::load(const char *path) {
    assert_msg(fd_ == -1, "load twice?");
    fd_ = open(path, O_RDONLY);
    assert_msg(fd_ != -1, "open file failed!");
    file_size_ = lseek(fd_, 0, SEEK_END);
    ptr_ = mmap(NULL, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    assert_msg(ptr_ != MAP_FAILED, "mmap failed!");
}

}
