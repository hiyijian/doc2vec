#ifndef WMD_H
#define WMD_H
#include "common_define.h"

class TaggedDocument;
class WeightedDocument;
class UnWeightedDocument;
struct knn_item_t;
class Doc2Vec;

class WMD
{
public:
  WMD(Doc2Vec * doc2vec);
  ~WMD();
  void train();
  void save(FILE * fout);
  void load(FILE * fin);
  real rwmd(WeightedDocument * src, UnWeightedDocument * target);
  void sent_knn_docs(TaggedDocument * doc, knn_item_t * knns, int k);
  void sent_knn_docs_ex(TaggedDocument * doc, knn_item_t * knns, int k);

private:
  void loadFromDoc2Vec();

public:
  UnWeightedDocument ** m_corpus;

  Doc2Vec * m_doc2vec;
  real * m_dis_vector;
  real * m_infer_vector;
  knn_item_t * m_doc2vec_knns;
};

#endif
