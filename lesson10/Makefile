#### a makefile that links and compiles all the libraries named .cpp and all the headers named .h ####

# Define the C++ compiler to use
CXX = mpic++

# Define the compiler flags to use
CXXFLAGS = -std=c++11 -Wall -Wextra -larmadillo -Wno-comment

# Define the name of the output executable
TARGET = main.exe

# Define the directories to search for source and header files
SRC_DIR = src
INCLUDE_DIR = include

# Define the source files to compile (all .cpp files in SRC_DIR)
SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp) main.cpp

# Define the object files to generate (same as SRC_FILES, but with .o extension)
OBJ_FILES = $(patsubst $(SRC_DIR)/%.cpp, $(SRC_DIR)/%.o, $(SRC_FILES))

# Define the header files to include (all .h files in INCLUDE_DIR)
INCLUDE_FILES = $(wildcard $(INCLUDE_DIR)/*.h)

# Define the rule to build the executable
$(TARGET): $(OBJ_FILES)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Define the rule to compile each source file into an object file
$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp $(INCLUDE_FILES)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Define a rule to clean up object and executable files
clean:
	rm -f $(SRC_DIR)/*.o $(TARGET)

### old and simple makefile
# CC = g++
# CFLAGS = -Wall -O3 --std=c++11 -larmadillo
#
# main.exe : main.o random.o
# 	$(CC) random.o main.o -o main.exe
# main.o : main.cpp
# 	$(CC) -c main.cpp -o main.o $(CFLAGS)
# random.o : ./src/generator/random.cpp ./src/generator/random.h
# 	$(CC) -c ./src/generator/random.cpp -o random.o $(CFLAGS)
# clean :
# 	rm *.o main.exe seed.out
