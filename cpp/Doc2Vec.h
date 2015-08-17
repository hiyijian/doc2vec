#ifndef DOC2VEC_H
#define DOC2VEC_H
#include "common_define.h"
#include <vector>

class TrainModelThread;
class NN;
class Vocabulary;
class WMD;
class TaggedBrownCorpus;
class TaggedDocument;
struct knn_item_t;

class Doc2Vec
{
friend class TrainModelThread;
friend class WMD;
friend class UnWeightedDocument;
friend class WeightedDocument;
public:
  Doc2Vec();
  ~Doc2Vec();
public:
  void train(const char * train_file,
    int dim, int cbow, int hs, int negtive,
    int iter, int window,
    real alpha, real sample,
    int min_count, int threads);
  long long dim();
  Vocabulary* wvocab();
  Vocabulary* dvocab();
  NN * nn();
  WMD * wmd();

public:
  real doc_likelihood(TaggedDocument * doc, int skip = -1);
  real context_likelihood(TaggedDocument * doc, int sentence_position);
  void infer_doc(TaggedDocument * doc, real * vector, int skip = -1);
  bool word_knn_words(const char * search, knn_item_t * knns, int k);
  bool doc_knn_docs(const char * search, knn_item_t * knns, int k);
  bool word_knn_docs(const char * search, knn_item_t * knns, int k);
  void sent_knn_words(TaggedDocument * doc, knn_item_t * knns, int k, real * infer_vector);
  void sent_knn_docs(TaggedDocument * doc, knn_item_t * knns, int k, real * infer_vector);
  real similarity(real * src, real * target);
  real distance(real * src, real * target);

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
  WMD * m_wmd;
  int m_cbow;
  int m_hs;
  int m_negtive;
  int m_window;
  real m_start_alpha; //fix lr
  real m_sample;
  int m_iter;

  //no need to flush to disk
  TaggedBrownCorpus * m_brown_corpus;
  real m_alpha; //working lr
  long long m_word_count_actual;
  real * m_expTable;
  int * m_negtive_sample_table;
  std::vector<TrainModelThread *> m_trainModelThreads;
};

struct knn_item_t
{
  char word[MAX_STRING];
  long long idx;
  real similarity;
};
void top_init(knn_item_t * knns, int k);
void top_collect(knn_item_t * knns, int k, long long idx, real similarity);
void top_sort(knn_item_t * knns, int k);
#endif
