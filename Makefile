CC=g++ -O3 -ggdb -std=c++20 -Wall -Wpedantic

traintorus: traintorus.o
	$(CC) traintorus.o -lm -o traintorus

traintorus.o: minigrad.hpp traintorus.cpp Makefile
	$(CC) -c traintorus.cpp

