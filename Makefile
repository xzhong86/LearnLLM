
ruby = ruby3

cpps = llm.cpp llm-utils.cpp nntools.cpp
hpps = llm-utils.hpp arrays.hpp nntools.hpp

run_llm: $(cpps) $(hpps)
	$(CXX) -Wall -std=c++23 -ferror-limit=3 -O -o $@ $(cpps)

test: run_llm
	./run_llm ../models/stories15M.bin ../llama2-c/tokenizer.bin

nntools.hpp: nntools.cpp gen-nntools-hpp.rb
	$(ruby) gen-nntools-hpp.rb $< > $@

arrays.hpp: gen-arrays-hpp.rb
	$(ruby) gen-arrays-hpp.rb > $@

