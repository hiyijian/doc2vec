#ifndef VOCAB_H
#define VOCAB_H

#include "common_define.h"

struct vocab_word_t
{
  long long cn; //frequency of word
  int *point; //Huffman tree(n leaf + n inner node, exclude root) path. (root, leaf], node index
  char *word; //word string
  char *code; //Huffman code. (root, leaf], 0/1 codes
  char codelen; //Hoffman code length
};

class Vocabulary
{
public:
  Vocabulary() {};
  Vocabulary(const char * train_file, int min_count = 5, bool doctag = false);
  ~Vocabulary();
public:
  long long searchVocab(const char *word);
  long long getVocabSize() {return m_vocab_size;}
  long long getTrainWords() {return m_train_words;};
  void save(FILE * fout);
  void load(FILE * fin);

private:
  void loadFromTrainFile(const char * train_file);
  long long addWordToVocab(const char *word);
  void sortVocab();
  void reduceVocab();
  void createHuffmanTree();

public:
  //first place is <s>, others sorted by its frequency reversely
  struct vocab_word_t* m_vocab;
  //size of vocab including <s>
  long long m_vocab_size;
  //total words of corpus. ie. sum up all frequency of words(exculude <s>)
  long long m_train_words;
private:
  long long m_vocab_capacity;
  //index: hash code of a word, value: vocab index of the word
  int *m_vocab_hash;
  int m_min_reduce;
  int m_min_count;
  bool m_doctag;
};

#endif
