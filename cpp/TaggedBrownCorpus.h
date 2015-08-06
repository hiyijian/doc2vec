#ifndef TAGGED_BROWN_CORPUS_H
#define TAGGED_BROWN_CORPUS_H

#include "common_define.h"

class TaggedDocument
{
public:
  TaggedDocument();
  ~TaggedDocument();
public:
  char * m_tag;
  char ** m_words;
  int m_word_num;
};

class TaggedBrownCorpus
{
public:
  TaggedBrownCorpus(const char * train_file, long long seek = 0, long long limit_doc = -1);
  ~TaggedBrownCorpus();

public:
  TaggedDocument * next();
  void rewind();
  long long getDocNum() {return m_doc_num;}
  long long tell() {return ftell(m_fin);}

private:
  int readWord(char *word);

private:
  FILE* m_fin;
  TaggedDocument m_doc;
  long long m_seek;
  long long m_doc_num;
  long long m_limit_doc;
};

#endif
