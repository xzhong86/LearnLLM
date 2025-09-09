
cpps = llm.cpp nntools.cpp
hpps = nntools.hpp

run_llm: $(cpps) $(hpps)
	$(CXX) -Wall -std=c++20 -ferror-limit=3 -o $@ $(cpps)

test: run_llm
	./run_llm ../models/stories15M.bin

nntools.hpp: nntools.cpp gen-nntools-hpp.rb
	ruby3 gen-nntools-hpp.rb $< > $@

