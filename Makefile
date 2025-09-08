
run_llm: llm.cpp
	$(CXX) -Wall -o $@ $^

test: run_llm
	./run_llm ../models/stories15M.bin

