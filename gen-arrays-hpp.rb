#!/usr/bin/env ruby

$base_array1d = """
template <typename T>
class Array1D {
    T * const data_;
    const int dim_;
public:
    Array1D(T* data, int dim) : data_(data), dim_(dim) {}
    Array1D(std::vector<T> &v) : data_(v.data()), dim_(v.size()) {}
    T*  data() const { return data; }
    int size() const { return dim_; }
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
  fail if dim < 2
  cls_name = "Array#{dim}D"
  cgen.puts "template <typename T>"
  cgen.put_block("class #{cls_name}", "{", "};") do
    cgen.puts "T * const data_;"
    cgen.puts "const int " + "dim{}_".map_sub_join(1..dim) + ";"
    cgen.puts "public:"
    cgen.puts("#{cls_name}(T *data, " + "int dim{}".map_sub_join(1..dim) +
              ") : data_(data), " + "dim{}_(dim{})".map_gsub_join(1..dim) +
              " {}")
    cgen.puts "long size() const { return 1L * " + "dim{}_".map_sub_join(1..dim, " * ") + "; }"
    1.upto(dim){ |d| cgen.puts "int d#{d}size() const { return dim#{d}_; }" }
    cgen.put_block("T& operator[](" + "int d{}".map_sub_join(1..dim) + ") const") do
      darr = 1.upto(dim).map{|d| (d..dim).to_a }
      sstr = ["d{}"] + ["dim{}_"] * (dim - 1)
      istr = darr.map{|a| a.zip(sstr).map{|n,s| s.sub('{}', n.to_s) }.join('*') }.join(' + ')
      cgen.puts "return data_[#{istr}];"
    end
    1.upto(dim-1) do |axd|  # ArrayXD
      dimr = 1..(dim - axd)
      cgen.put_block("Array#{axd}D<T> operator[](" + "int d{}".map_sub_join(dimr) + ") const") do
        darr = 1.upto(dim).map{|d| (d..dim).to_a }.slice(0, dim - axd)
        sstr = ["d{}"] + ["dim{}_"] * (dim - 1)
        istr = darr.map{|a| a.zip(sstr).map{|n,s| s.sub('{}', n.to_s) }.join('*') }.join(' + ')
        sdimr = (dimr.max + 1) .. dim
        cgen.puts "return Array#{axd}D<T>(data_ + #{istr}, " + "dim{}_".map_sub_join(sdimr) + ");"
      end
    end
  end
end

puts "#pragma once"
puts "namespace nn {"
puts $base_array1d
cgen = CodeGen.new
gen_ArrayXD(cgen, 2)
gen_ArrayXD(cgen, 3)
gen_ArrayXD(cgen, 4)
#gen_ArrayXD(cgen, 5)
puts "}"
