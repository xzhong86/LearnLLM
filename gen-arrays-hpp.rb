#!/usr/bin/env ruby

$base_array1d = """
template <typename T>
class Array1D {
    T * data_;
    int dim_;
public:
    Array1D() : data_(nullptr), dim_(0) {}
    Array1D(T* data, int dim) : data_(data), dim_(dim) {}
    T*  data() const { return data_; }
    int size() const { return dim_; }
    void copy(const Array1D<T> &from) {
         assert_msg(from.dim_ == dim_, \"mismatch\");
         for (int i = 0; i < dim_; i++) data_[i] = from.data_[i];
    }
    T& operator[](int i) const { return data_[i]; }
};
"""

class String
  def map_sub_join(values, sep = ", ")
    values.map{ |v| self.sub('{}', v.to_s) }.join(sep)
  end
  def map_gsub_join(values, sep = ", ")
    values.map{ |v| self.gsub('{}', v.to_s) }.join(sep)
  end
end

class CodeGen
  def initialize()
    @indent = 0
  end
  def indent_spc()
    "    " * @indent
  end
  alias io_print print
  alias io_puts  puts
  def puts(str)
    io_puts indent_spc() + str
  end
  def put_block(prefix, sob = "{", eob = "}")
    puts prefix + " " + sob
    @indent += 1
    yield
    @indent -= 1
    puts eob
  end
end

def gen_ArrayXD(cgen, dim)
  fail "bad dim" if dim < 2
  do_check = false  # manually open this
  def check_dimr(cgen, dr)
    dr.each{ |d| cgen.puts "assert_msg(0 <= d#{d} && d#{d} < dim#{d}_, \"overmax\");" }
  end

  cls_name = "Array#{dim}D"
  cgen.puts "template <typename T>"
  cgen.put_block("class #{cls_name}", "{", "};") do
    cgen.puts "T * data_;"
    cgen.puts "int " + "dim{}_".map_sub_join(1..dim) + ";"
    cgen.puts "public:"
    cgen.puts("#{cls_name}() : data_(nullptr), " + "dim{}_(0)".map_sub_join(1..dim) + " {}")
    cgen.puts("#{cls_name}(T *data, " + "int dim{}".map_sub_join(1..dim) +
              ") : data_(data), " + "dim{}_(dim{})".map_gsub_join(1..dim) +
              " {}")
    cgen.puts "long size() const { return 1L * " + "dim{}_".map_sub_join(1..dim, " * ") + "; }"
    cgen.puts "T*   raw()  const { return data_; }"
    1.upto(dim){ |d| cgen.puts "int d#{d}size() const { return dim#{d}_; }" }
    cgen.put_block("T& operator[](" + "int d{}".map_sub_join(1..dim) + ") const") do
      check_dimr(cgen, 1..dim) if do_check
      darr = 1.upto(dim).map{|d| (d..dim).to_a }
      sstr = ["d{}"] + ["dim{}_"] * (dim - 1)
      istr = darr.map{|a| a.zip(sstr).map{|n,s| s.sub('{}', n.to_s) }.join('*') }.join(' + ')
      cgen.puts "return data_[#{istr}];"
    end
    1.upto(dim-1) do |axd|  # ArrayXD
      dimr = 1..(dim - axd)
      cgen.put_block("Array#{axd}D<T> operator[](" + "int d{}".map_sub_join(dimr) + ") const") do
        check_dimr(cgen, dimr) if do_check
        darr = 1.upto(dim).map{|d| (d..dim).to_a }.slice(0, dim - axd)
        sstr = ["d{}"] + ["dim{}_"] * (dim - 1)
        istr = darr.map{|a| a.zip(sstr).map{|n,s| s.sub('{}', n.to_s) }.join('*') }.join(' + ')
        sdimr = (dimr.max + 1) .. dim
        cgen.puts "return Array#{axd}D<T>(data_ + #{istr}, " + "dim{}_".map_sub_join(sdimr) + ");"
      end
    end
  end
end

def gen_ArrayPool(cgen, max_dim)
  cgen.puts "template <typename T>"
  cgen.put_block("class ArrayPool", "{", "};") do
    lines = """
    |  T * pool_;
    |  T * ptr_ = nullptr;
    |  long size_ = 0;
    |  bool no_del_ = false;
    |public:
    |  ArrayPool() : pool_(nullptr) {}
    |  ArrayPool(long size) : pool_(new T[size]), size_(size) {
    |      ptr_ = pool_;
    |  }
    |  ~ArrayPool() { if (!no_del_) delete [] pool_; }
    |  void alloc(long size) {
    |      assert_msg(pool_ == nullptr, \"not empty\");
    |      pool_ = new T[size]; size_ = size; ptr_ = pool_;
    |  }
    |  void useSpace(T *p, long size) {
    |      assert_msg(pool_ == nullptr, \"not empty\");
    |      pool_ = p; size_ = size; ptr_ = pool_; no_del_ = true;
    |  }
    |"""
    lines.lines.map{|l| l.chomp.sub(/^\s+\|/,'') }.each{|l| cgen.puts l if l.length > 0 }
    (1..max_dim).each do |dim|
      cgen.put_block("Array#{dim}D<T> alloc#{dim}D(" + "int d{}".map_sub_join(1..dim) + ")") do
        cgen.puts "auto a = Array#{dim}D<T>(ptr_, " + "d{}".map_sub_join(1..dim) + ");"
        cgen.puts "ptr_ += a.size();"
        cgen.puts "assert_msg(ptr_ <= pool_ + size_, \"overflow\");"
        cgen.puts "return a;"
      end
    end
  end
end

# main

puts "#pragma once"
puts "#include \"llm-utils.hpp\""
puts "namespace nn {"
puts "using namespace llm;"
puts $base_array1d
cgen = CodeGen.new
max_dim = 4
(2..max_dim).each{|d| gen_ArrayXD(cgen, d) }
gen_ArrayPool(cgen, max_dim)
puts "}"
