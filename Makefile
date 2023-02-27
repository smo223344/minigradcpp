CC=g++ -O3 -ggdb -std=c++20 -Wall -Wpedantic

all: traintorus traindots

traindots: traindots.o
	$(CC) traindots.o -lm -o traindots

traindots.o: minigrad.hpp traindots.cpp Makefile
	$(CC) -c traindots.cpp

traintorus: traintorus.o
	$(CC) traintorus.o -lm -o traintorus

traintorus.o: minigrad.hpp traintorus.cpp Makefile
	$(CC) -c traintorus.cpp


