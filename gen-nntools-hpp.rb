#!/usr/bin/env ruby

require 'optparse'
require 'ostruct'

FuncArg = Struct.new :type, :name

def read_cppfile(file)
  info = OpenStruct.new(namespace: nil, funcs: [])
  IO.foreach(file) do |line|
    if line =~ /^namespace (\S+)/
      fail "multi namespace?" if info.namespace
      info.namespace = $1
    elsif line =~ /^(\w+)\s+(\w+)\((.*)\)/
      ret, func, args = $1, $2, $3.split(/\s*,\s*/)
      args.map! do |arg|
        flds = arg.split(/\s+/)
        fail if flds.size != 2
        type, name = *flds
        if name =~ /^(\*+)/
          type << $1
          name.delete_prefix($1)
        end
        FuncArg.new(type, name)
      end
      info.funcs << OpenStruct.new(ret: ret, name: func, args: args)
    end
  end
  info
end

def gen_func_decl(func, args=nil)
  args ||= func.args
  args_str = args.map{ |a| a.type + ' ' + a.name }.join(", ")
  "#{func.ret} #{func.name}(#{args_str})"
end
def gen_conv_func(func, args)
  has_vec = args.count{|a| a.type.index("vector") } > 0
  if has_vec
    puts "inline " + gen_func_decl(func, args) + "{"
    ret_pfx = func.ret == "void" ? "" : "return "
    arg_str = args.map do |arg|
      if arg.type =~ /vector/
        arg.name + ".data()"
      else
        arg.name
      end
    end.join(", ")
    puts "    #{ret_pfx}#{func.name}(#{arg_str});"
    puts "}"
  end
end

def gen_header(info)
  puts "#include <vector>"
  puts "namespace #{info.namespace} {" if info.namespace
  puts "using std::vector;"
  info.funcs.each do |func|
    puts gen_func_decl(func) + ';'
  end
  info.funcs.each do |func|
    args_psb = func.args.map do |arg|
      if arg.type == "float*"
        [arg, FuncArg.new("vector<float>&", arg.name)]
      else
        [arg]
      end
    end
    Enumerator.product(*args_psb).each do |args|
      gen_conv_func(func, args)
    end
  end
  puts "} // namespace #{info.namespace}" if info.namespace
end

# main
opts = OptionParser.new do |op|
  op.banner = "%s [options] nntools.cpp" % File.basename($0)
end.parse!

cppfile = ARGV.shift

info = read_cppfile(cppfile)
gen_header(info)
