#ifndef DOC2VEC_H
#define DOC2VEC_H
#include "common_define.h"
#include "NN.h"
#include "Vocab.h"
#include "TaggedBrownCorpus.h"
#include <vector>

class TrainModelThread;
struct knn_item_t;

class Doc2Vec
{
friend class TrainModelThread;
public:
  Doc2Vec();
  ~Doc2Vec();
public:
  void train(const char * train_file,
    int dim, int cbow, int hs, int negtive,
    int iter, int window,
    real alpha, real sample,
    int min_count, int threads);
public:
  void infer_doc(TaggedDocument * doc, real * vector, int iter = 50);
  bool word_knn_words(const char * search, knn_item_t * knns, int k);
  bool doc_knn_docs(const char * search, knn_item_t * knns, int k);
  bool word_knn_docs(const char * search, knn_item_t * knns, int k);
  void sent_knn_words(TaggedDocument * doc, knn_item_t * knns, int k, real * infer_vector);
  void sent_knn_docs(TaggedDocument * doc, knn_item_t * knns, int k, real * infer_vector);
  real similarity(real * src, real * target);

public:
  void save(FILE * fout);
  void load(FILE * fin);
private:
  void initExpTable();
  void initNegTable();
  void initTrainModelThreads(const char * train_file, int threads, int iter);
  bool obj_knn_objs(const char * search, real* src,
    bool search_is_word, bool target_is_word,
    knn_item_t * knns, int k);

private:
  Vocabulary * m_word_vocab;
  Vocabulary * m_doc_vocab;
  NN * m_nn;
  int m_cbow;
  int m_hs;
  int m_negtive;
  int m_window;
  real m_start_alpha; //fix lr
  real m_sample;

  //no need to flush to disk
  real m_alpha; //working lr
  long long m_word_count_actual;
  real * m_expTable;
  int * m_negtive_sample_table;
  std::vector<TrainModelThread *> m_trainModelThreads;
};

class TrainModelThread
{
friend class Doc2Vec;
public:
  TrainModelThread(long long id, Doc2Vec * doc2vec,
    TaggedBrownCorpus* sub_corpus, int iter, bool infer = false);
  ~TrainModelThread();
public:
  void train();

private:
  void updateLR();
  void buildDocument(TaggedDocument * doc);
  void trainSampleCbow(long long central, long long context_start, long long context_end);
  void trainPairSg(long long central_word, real * context);
  void trainSampleSg(long long central, long long context_start, long long context_end);
  void trainDocument();
  bool down_sample(long long cn);
  long long negtive_sample();

private:
  long long m_id;
  Doc2Vec * m_doc2vec;
  TaggedBrownCorpus* m_corpus;
  int m_iter;
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

struct knn_item_t
{
  char word[MAX_STRING];
  real similarity;
};

#endif
