GCC=g++
MALLOC=/usr/include/malloc
INCLUDEDIR=-I$(MALLOC)
LIBDIR=-lm -pthread
CPPFLAGS = -O3 -Wall -march=native -funroll-loops -Wno-unused-result -Wno-int-to-pointer-cast
TARGET = train
OBJS = train.o Doc2Vec.o Vocab.o NN.o TaggedBrownCorpus.o

all: ${TARGET}

clean:
	rm -f *.o $(TARGET)

${TARGET} : $(OBJS)
	$(GCC) $(CPPFLAGS) $^ $(LIBDIR) -o $@

%.o	: %.cpp
	$(GCC) $(CPPFLAGS) -c $< -o $@ $(INCLUDEDIR)
