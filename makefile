# Declaration of variables
CC = g++-13
CC_FLAGS = -O3 -frename-registers -funroll-loops -std=c++11
CC_SIMPLE = -std=c++17 -O2

CC_BASIC = -O2 -Wall -std=c++17 -Wno-unused-const-variable
CC_OLD = -Wall -DLOCAL_PROJECT -Wextra -std=c++20 -Wmaybe-uninitialized -Wuninitialized -Wno-unused-parameter -Wno-unused-const-variable -Wshadow -Wformat=2 -Wfloat-equal -Wconversion -Wno-float-conversion -Wno-sign-conversion -pedantic -Wshift-overflow -Wcast-qual -Wcast-align -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC -fno-sanitize-recover=all -fstack-protector
# CC_FULL = -Wall -DLOCAL_PROJECT -Wextra -std=c++20 -Wmaybe-uninitialized -Wuninitialized -Wno-unused-parameter -Wno-unused-const-variable -Wshadow -Wformat=2 -Wfloat-equal -Wconversion -Wno-float-conversion -Wno-sign-conversion -pedantic -Wshift-overflow=2 -Wduplicated-cond -Wcast-qual -Wcast-align -Wlogical-op  -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC -fno-sanitize-recover=all -fstack-protector -g
CC_FULL = -Wall -DLOCAL_PROJECT -Wextra -std=c++20 -Wmaybe-uninitialized -Wuninitialized -Wno-unused-parameter -Wno-unused-const-variable -Wshadow -Wformat=2 -Wfloat-equal -Wconversion -Wno-float-conversion -Wno-sign-conversion -pedantic -Wshift-overflow=2 -Wduplicated-cond -Wcast-qual -Wcast-align -Wlogical-op  -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC -fno-sanitize-recover=all -fstack-protector -g -fmax-errors=5
CC_TEST = -Wall -Wextra -pedantic -std=c++11 -O2 -Wshadow -Wformat=2 -Wfloat-equal -Wconversion -Wlogical-op -Wshift-overflow=2 -Wduplicated-cond -Wcast-qual -Wcast-align -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC -D_FORTIFY_SOURCE=2 -fsanitize=address -fsanitize=undefined -fno-sanitize-recover -fstack-protector
CC_DEBUG = -std=c++17 -Wshadow -Wall -fsanitize=address -fsanitize=undefined -Wconversion -D_GLIBCXX_DEBUG -g

CCF2 = -O3 -fopenmp -D_GLIBCXX_PARALLEL -frename-registers -std=c++11
CCF3 = -O3 -fopenmp -D_GLIBCXX_PARALLEL -frename-registers -fprofile-use -std=c++11

#g++ -o P67356.cc P67356.exe-std=c++17
# https://codeforces.com/blog/entry/15547

# File names
SRC = $(wildcard *.cpp)
#EXEC = $(SRC:.cc=.exe)
PROGS = $(patsubst %.cpp,%.exe,$(SRC))

all: $(PROGS)

# Main target
%.exe: %.cpp
	$(CC) -o $@ $< $(CC_FULL) -D LOCAL_RUN

# To remove generated files
clean:
	rm -f *.exe

.PHONY: clean all
