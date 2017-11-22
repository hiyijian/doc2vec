#ifndef NN_H
#define NN_H
#include "common_define.h"

class NN
{
public:
  NN() : m_syn0(NULL), m_dsyn0(NULL), m_syn1(NULL), m_syn1neg(NULL),m_syn0norm(NULL), m_dsyn0norm(NULL)  {}
  NN(long long vocab_size, long long corpus_size, long long dim, int hs, int negtive);
  ~NN();
public:
  void save(FILE * fout);
  void load(FILE * fin);
  void norm();
public:
  int m_hs;
  int m_negtive;
  real *m_syn0, *m_dsyn0, *m_syn1, *m_syn1neg;
  long long m_vocab_size, m_corpus_size, m_dim;
  //no need to flush to disk
  real * m_syn0norm, * m_dsyn0norm;
};

#endif
