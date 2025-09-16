
#pragma once

#include <iostream>
#include <source_location>

namespace llm {

inline void throw_error(const char *msg) {
    throw std::runtime_error(msg);
}

template <typename T>
void assert_msg(T cond, const char *msg,
            const std::source_location& loc = std::source_location::current())
{
    if (!(cond)) {
        std::cerr << "Assert failed: '" << msg << "' at " << loc.file_name()
                  << ":" << loc.line() << " " << loc.function_name()
                  << std::endl;
        throw_error("assert failed.");
    }
}

#ifdef assert
# warn "should not use c-style assert"
# undefine assert
#endif
#define assert(c)   assert_msg((c), #c)


template <typename T>
inline void read_from(T *p, std::istream &is) {
    is.read(reinterpret_cast<char*>(p), sizeof(T));
}
template <typename T>
inline void read_from(T *p, int n, std::istream &is) {
    is.read(reinterpret_cast<char*>(p), sizeof(T) * n);
}


class OSMemMap {
    int fd_ = -1;
    size_t file_size_ = 0;
    void *ptr_ = nullptr;
public:
    OSMemMap() {}
    OSMemMap(const char *path) { load(path); }
    ~OSMemMap();

    void load(const char *path);
    size_t getMemSize() const { return file_size_; }

    template <typename T>
    T *getPtr(int offset_bytes) const {
        assert_msg(offset_bytes % sizeof(T) == 0, "offset misalign");
        assert_msg(offset_bytes >= 0 && offset_bytes < file_size_, "bad offset");
        return static_cast<T*>(ptr_) + offset_bytes / sizeof(T);
    }
};

} // namespace llm
