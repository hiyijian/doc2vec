#ifndef NN_H
#define NN_H
#include "common_define.h"

class NN
{
public:
  NN() {}
  NN(long long vocab_size, long long corpus_size, long long dim);
  ~NN();
public:
  void save(FILE * fout);
  void load(FILE * fin);
  void norm();
public:
  real *m_syn0, *m_dsyn0, *m_syn1, *m_syn1neg;
  long long m_vocab_size, m_corpus_size, m_dim;
  //no need to flush to disk
  real * m_syn0norm, * m_dsyn0norm;
};

#endif
