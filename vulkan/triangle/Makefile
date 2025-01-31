CFLAGS = -std=c++17
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Linux)
	# OSFLAG += -D LINUX
	LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi
endif
ifeq ($(UNAME_S),Darwin)
	# OSFLAG += -D OSX
	LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11
endif



all: main.cpp
	g++ $(CFLAGS) -o a.out main.cpp $(LDFLAGS)

test: texture_mapping.cpp
	g++ $(CFLAGS) -o a.out texture_mapping.cpp $(LDFLAGS)



