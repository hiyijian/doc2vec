#ifndef TRAIN_MODEL_THREAD_H
#define TRAIN_MODEL_THREAD_H
#include "common_define.h"

class Doc2Vec;
class TaggedBrownCorpus;
class TaggedDocument;

class TrainModelThread
{
friend class Doc2Vec;
public:
  TrainModelThread(long long id, Doc2Vec * doc2vec,
    TaggedBrownCorpus* sub_corpus, bool infer = false);
  ~TrainModelThread();
public:
  void train();

private:
  void updateLR();
  void buildDocument(TaggedDocument * doc, int skip = -1);
  void trainSampleCbow(long long central, long long context_start, long long context_end);
  void trainPairSg(long long central_word, real * context);
  void trainSampleSg(long long central, long long context_start, long long context_end);
  void trainDocument();
  bool down_sample(long long cn);
  long long negtive_sample();
  real doc_likelihood();
  real context_likelihood(long long sentence_position);
  real likelihoodPair(long long central, real * context_vector);


private:
  long long m_id;
  Doc2Vec * m_doc2vec;
  TaggedBrownCorpus* m_corpus;
  bool m_infer;

  clock_t m_start;
  unsigned long long m_next_random;

  long long m_sen[MAX_SENTENCE_LENGTH];
  long long m_sentence_length;
  long long m_sen_nosample[MAX_SENTENCE_LENGTH];
  long long m_sentence_nosample_length;
  real * m_doc_vector;
  long long m_word_count;
  long long m_last_word_count;
  real *m_neu1;
  real *m_neu1e;
};

#endif
