# Build: first compile the model, then this program
#   nnc compile ../model.nnl --emit lib -o ../libsimple_mlp.a
#   nnc compile ../model.nnl --emit header -o ../simple_mlp.h
#   make

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
LDFLAGS = -L.. -lsimple_mlp -lm

all: infer

infer: main.cpp
	$(CXX) $(CXXFLAGS) -I.. -o $@ $< $(LDFLAGS)

clean:
	rm -f infer

.PHONY: all clean
