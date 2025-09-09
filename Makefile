
run_llm: llm.cpp
	$(CXX) -Wall -std=c++20 -o $@ $^

test: run_llm
	./run_llm ../models/stories15M.bin

