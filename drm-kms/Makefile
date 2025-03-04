CC := gcc
CFLAGS := -Wall -Wextra -g $(shell pkg-config --cflags libdrm)
LDFLAGS := $(shell pkg-config --libs libdrm)

all: main

main: main.o
	$(CC) main.o -o main $(LDFLAGS)

main.o: main.c
	$(CC) -c main.c $(CFLAGS)

clean:
	rm main main.o
