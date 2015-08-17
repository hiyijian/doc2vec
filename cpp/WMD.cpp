#include "WMD.h"
#include "TaggedBrownCorpus.h"
#include "Vocab.h"
#include "NN.h"
#include "Doc2Vec.h"

// //////////////WMD/////////////////////////////
WMD::WMD(Doc2Vec * doc2vec): m_corpus(NULL), m_doc2vec(doc2vec),
  m_dis_vector(NULL), m_infer_vector(NULL),
  m_doc2vec_knns(NULL)
{
  m_corpus = new UnWeightedDocument*[m_doc2vec->m_nn->m_corpus_size];
  for(long long a = 0; a < m_doc2vec->m_nn->m_corpus_size; a++) m_corpus[a] = NULL;
  posix_memalign((void **)&m_dis_vector, 128, MAX_SENTENCE_LENGTH * sizeof(real));
  posix_memalign((void **)&m_infer_vector, 128, m_doc2vec->m_nn->m_dim * sizeof(real));
  m_doc2vec_knns = new knn_item_t[MAX_DOC2VEC_KNN];
}

WMD::~WMD()
{
  long long a;
  if(m_corpus) for(a = 0; a < m_doc2vec->m_nn->m_corpus_size; a++) if(m_corpus[a]) delete m_corpus[a];
  if(m_corpus) delete [] m_corpus;
  if(m_dis_vector) free(m_dis_vector);
  if(m_infer_vector) free(m_infer_vector);
  if(m_doc2vec_knns) delete [] m_doc2vec_knns;
}

void WMD::train()
{
  loadFromDoc2Vec();
}

void WMD::save(FILE * fout)
{
  for(long long a = 0; a < m_doc2vec->m_nn->m_corpus_size; a++) m_corpus[a]->save(fout);
}

void WMD::load(FILE * fin)
{
  m_corpus = new UnWeightedDocument*[m_doc2vec->m_nn->m_corpus_size];
  for(long long a = 0; a < m_doc2vec->m_nn->m_corpus_size; a++)
  {
    m_corpus[a] = new UnWeightedDocument();
    m_corpus[a]->load(fin);
    if(m_corpus[a]->m_word_num <= 0){
      delete m_corpus[a];
      m_corpus[a] = NULL;
    }
  }
}

void WMD::loadFromDoc2Vec()
{
  long long doc_idx;
  TaggedDocument * doc = NULL;
  m_doc2vec->m_brown_corpus->rewind();
  while((doc = m_doc2vec->m_brown_corpus->next()) != NULL)
  {
    doc_idx = m_doc2vec->m_doc_vocab->searchVocab(doc->m_tag);
    if(doc_idx == -1) continue;
    m_corpus[doc_idx] = new UnWeightedDocument(m_doc2vec, doc);
  }
}

void WMD::sent_knn_docs(TaggedDocument * doc, knn_item_t * knns, int k)
{
  long long b, c;
  UnWeightedDocument * target;
  WeightedDocument src(m_doc2vec, doc);
  top_init(knns, k);
  for(b = 1, c = 0; b < m_doc2vec->m_nn->m_corpus_size; b++)
  {
    target = m_corpus[b];
    if(target)
    {
      if(c < k) {
        knns[c].similarity = -rwmd(&src, target);
        knns[c].idx = b;
        c++;
        if(c == k) top_init(knns, k);
      }
      else top_collect(knns, k, b, -rwmd(&src, target));
    }
  }
  top_sort(knns, k);
  for(b = 0; b < k; b++) strcpy(knns[b].word, m_doc2vec->m_doc_vocab->m_vocab[knns[b].idx].word);
}

void WMD::sent_knn_docs_ex(TaggedDocument * doc, knn_item_t * knns, int k)
{
  m_doc2vec->sent_knn_docs(doc, m_doc2vec_knns, MAX_DOC2VEC_KNN, m_infer_vector);
  long long b, c, idx;
  UnWeightedDocument * target;
  WeightedDocument src(m_doc2vec, doc);
  top_init(knns, k);
  for(b = 0, c = 0; b < MAX_DOC2VEC_KNN; b++)
  {
    idx = m_doc2vec_knns[b].idx;
    target = m_corpus[idx];
    if(target)
    {
      if(c < k) {
        knns[c].similarity = -rwmd(&src, target);
        knns[c].idx = idx;
        c++;
        if(c == k) top_init(knns, k);
      }
      else top_collect(knns, k, idx, -rwmd(&src, target));
    }
  }
  top_sort(knns, k);
  for(b = 0; b < k; b++) strcpy(knns[b].word, m_doc2vec->m_doc_vocab->m_vocab[knns[b].idx].word);
}

real WMD::rwmd(WeightedDocument * src, UnWeightedDocument * target)
{
  if(src->m_word_num <= 0 || target->m_word_num <= 0) return (std::numeric_limits<double>::max)();
  int a, b;
  real score, l1 = 0;
  real * syn0norm = m_doc2vec->m_nn->m_syn0norm;
  long long dim = m_doc2vec->m_nn->m_dim;
  for(a = 0; a < src->m_word_num; a++) m_dis_vector[a] = (std::numeric_limits<double>::max)();
  for(a = 0; a < src->m_word_num; a++)
    for(b = 0; b < target->m_word_num; b++) {
      score = m_doc2vec->distance(&(syn0norm[src->m_words_idx[a] * dim]), &(syn0norm[target->m_words_idx[b] * dim]));
      m_dis_vector[a] = MIN(m_dis_vector[a], score);
    }
  for(a = 0; a < src->m_word_num; a++) l1 += m_dis_vector[a] * src->m_words_wei[a];
  return l1;
}
