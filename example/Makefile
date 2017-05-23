CXX = g++
FLAGS = -std=c++1z -O2 -I../include
LIBFLAG = `pkg-config --libs opencv`

example:
	$(CXX) $(FLAGS) example.cpp -o example $(LIBFLAG)

all: example
