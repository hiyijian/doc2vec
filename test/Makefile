GCC=g++
INCLUDEDIR=-I./ -I/opt/gtest/include -I../cpp
LIBDIR=-L/opt/gtest/lib -lgtest -lgtest_main -lpthread
CPPFLAGS = -O3 -Wall
TARGET = test
TEST_OBJS = ../cpp/Doc2Vec.o ../cpp/TrainModelThread.o ../cpp/NN.o ../cpp/TaggedBrownCorpus.o ../cpp/Vocab.o ../cpp/WMD.o
OBJS = test.o TestKeyword.o TestTrain.o TestSimilar.o $(TEST_OBJS)

all: ${TARGET}

clean:
	rm -f *.o $(TARGET)

${TARGET} : $(OBJS)
	$(GCC) $(CPPFLAGS) $^ $(LIBDIR) -o $@

%.o	: %.cpp
	$(GCC) $(CPPFLAGS) -c $< -o $@ $(INCLUDEDIR)
