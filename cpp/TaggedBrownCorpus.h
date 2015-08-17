#ifndef TAGGED_BROWN_CORPUS_H
#define TAGGED_BROWN_CORPUS_H

#include "common_define.h"
class Doc2Vec;
//==================TaggedDocument============================
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
//==================TaggedBrownCorpus============================
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

//==================UnWeightedDocument============================
class Doc2Vec;

class UnWeightedDocument
{
public:
  UnWeightedDocument();
  UnWeightedDocument(Doc2Vec * doc2vec, TaggedDocument * doc);
  virtual ~UnWeightedDocument();

public:
  void save(FILE * fout);
  void load(FILE * fin);
public:
  long long * m_words_idx;
  int m_word_num;
};
//==================WeightedDocument============================
class WeightedDocument : public UnWeightedDocument
{
public:
  WeightedDocument(Doc2Vec * doc2vec, TaggedDocument * doc);
  virtual ~WeightedDocument();

public:
  real * m_words_wei;
};

#endif
