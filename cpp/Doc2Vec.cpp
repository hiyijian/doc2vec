#include "Doc2Vec.h"
#include "NN.h"
#include "Vocab.h"
#include "WMD.h"
#include "TrainModelThread.h"
#include "TaggedBrownCorpus.h"

static void * trainModelThread(void * params);

void * trainModelThread(void * params)
{
  TrainModelThread * tparams = (TrainModelThread *)params;
  tparams->train();
  return NULL;
}

/////==============================DOC2VEC========================
Doc2Vec::Doc2Vec(): m_word_vocab(NULL), m_doc_vocab(NULL), m_nn(NULL), m_wmd(NULL),
  m_brown_corpus(NULL), m_expTable(NULL), m_negtive_sample_table(NULL)
{
  initExpTable();
}

Doc2Vec::~Doc2Vec()
{
  if(m_word_vocab) delete m_word_vocab;
  if(m_doc_vocab) delete m_doc_vocab;
  if(m_nn) delete m_nn;
  if(m_wmd) delete m_wmd;
  if(m_brown_corpus) delete m_brown_corpus;
  if(m_expTable) free(m_expTable);
  if(m_negtive_sample_table) free(m_negtive_sample_table);
  for(size_t i =  0; i < m_trainModelThreads.size(); i++) delete m_trainModelThreads[i];
}

void Doc2Vec::initExpTable()
{
  m_expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (int i = 0; i < EXP_TABLE_SIZE; i++)
  {
    m_expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    m_expTable[i] = m_expTable[i] / (m_expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
}

void Doc2Vec::initNegTable()
{
  int a, i;
  long long train_words_pow = 0;
  real d1, power = 0.75;
  m_negtive_sample_table = (int *)malloc(negtive_sample_table_size * sizeof(int));
  for (a = 0; a < m_word_vocab->m_vocab_size; a++) train_words_pow += pow(m_word_vocab->m_vocab[a].cn, power);
  i = 0;
  d1 = pow(m_word_vocab->m_vocab[i].cn, power) / (real)train_words_pow;
  for (a = 0; a < negtive_sample_table_size; a++) {
    m_negtive_sample_table[a] = i;
    if (a / (real)negtive_sample_table_size > d1) {
      i++;
      d1 += pow(m_word_vocab->m_vocab[i].cn, power) / (real)train_words_pow;
    }
    if (i >= m_word_vocab->m_vocab_size) i = m_word_vocab->m_vocab_size - 1;
  }
}

void Doc2Vec::train(const char * train_file,
  int dim, int cbow, int hs, int negtive,
  int iter, int window,
  real alpha, real sample,
  int min_count, int threads)
{
  printf("Starting training using file %s\n", train_file);
  m_cbow = cbow;
  m_hs = hs;
  m_negtive = negtive;
  m_window = window;
  m_start_alpha = alpha;
  m_sample = sample;
  m_iter = iter;

  m_word_vocab = new Vocabulary(train_file, min_count);
  m_doc_vocab = new Vocabulary(train_file, 1, true);
  m_nn = new NN(m_word_vocab->m_vocab_size, m_doc_vocab->m_vocab_size, dim, hs, negtive);
  if(m_negtive > 0) initNegTable();

  m_brown_corpus = new TaggedBrownCorpus(train_file);
  m_alpha = alpha;
  m_word_count_actual = 0;
  initTrainModelThreads(train_file, threads, iter);

  printf("Train with %d threads\n", (int)m_trainModelThreads.size());
  pthread_t *pt = (pthread_t *)malloc(m_trainModelThreads.size() * sizeof(pthread_t));
  for(size_t a = 0; a < m_trainModelThreads.size(); a++) {
    pthread_create(&pt[a], NULL, trainModelThread, (void *)m_trainModelThreads[a]);
  }
  for (size_t a = 0; a < m_trainModelThreads.size(); a++) pthread_join(pt[a], NULL);
  free(pt);

  m_nn->norm();
  m_wmd = new WMD(this);
  m_wmd->train();
}

void Doc2Vec::initTrainModelThreads(const char * train_file, int threads, int iter)
{
  long long limit = m_doc_vocab->m_vocab_size / threads;
  long long sub_size = 0;
  long long tell = 0;
  TaggedBrownCorpus brown_corpus(train_file);
  TaggedBrownCorpus * sub_c = NULL;
  TrainModelThread * model_thread = NULL;
  TaggedDocument * doc = NULL;
  while((doc = brown_corpus.next()) != NULL)
  {
    sub_size++;
    if(sub_size >= limit)
    {
        sub_c = new TaggedBrownCorpus(train_file, tell, sub_size);
        model_thread = new TrainModelThread(m_trainModelThreads.size(), this, sub_c, false);
        m_trainModelThreads.push_back(model_thread);
        tell = brown_corpus.tell();
        sub_size = 0;
    }
  }
  if(m_trainModelThreads.size() < size_t(threads))
  {
    sub_c = new TaggedBrownCorpus(train_file, tell, -1);
    model_thread = new TrainModelThread(m_trainModelThreads.size(), this, sub_c, false);
    m_trainModelThreads.push_back(model_thread);
  }
  printf("corpus size: %lld\n", m_doc_vocab->m_vocab_size - 1);
  return;
}

bool Doc2Vec::obj_knn_objs(const char * search, real * src,
  bool search_is_word, bool target_is_word,
  knn_item_t * knns, int k)
{
  long long a = -1, b, c, target_size;
  real * search_vectors, * target, * target_vectors;
  Vocabulary * search_vocab, * target_vocab;
  search_vocab = search_is_word ? m_word_vocab : m_doc_vocab;
  search_vectors = search_is_word ? m_nn->m_syn0norm : m_nn->m_dsyn0norm;
  target_vectors = target_is_word ? m_nn->m_syn0norm : m_nn->m_dsyn0norm;
  target_size = target_is_word ? m_nn->m_vocab_size : m_nn->m_corpus_size;
  target_vocab = target_is_word ? m_word_vocab : m_doc_vocab;
  if(!src) {
    a = search_vocab->searchVocab(search);
    if(a < 0) return false;
    src = &(search_vectors[a * m_nn->m_dim]);
  }
  for(b = 0, c = 0; b < target_size; b++)
  {
    if(search_is_word == target_is_word && a == b) continue;
    target = &(target_vectors[b * m_nn->m_dim]);
    if(c < k){
      knns[c].similarity = similarity(src, target);
      knns[c].idx = b;
      c++;
      if(c == k) top_init(knns, k);
    }
    else top_collect(knns, k, b, similarity(src, target));
  }
  top_sort(knns, k);
  for(b = 0; b < k; b++) strcpy(knns[b].word, target_vocab->m_vocab[knns[b].idx].word);
  return true;
}

bool Doc2Vec::word_knn_words(const char * search, knn_item_t * knns, int k)
{
  return obj_knn_objs(search, NULL, true, true, knns, k);
}

bool Doc2Vec::doc_knn_docs(const char * search, knn_item_t * knns, int k)
{
  return obj_knn_objs(search, NULL, false, false, knns, k);
}

bool Doc2Vec::word_knn_docs(const char * search, knn_item_t * knns, int k)
{
  return obj_knn_objs(search, NULL, true, false, knns, k);
}

void Doc2Vec::sent_knn_words(TaggedDocument * doc, knn_item_t * knns, int k, real * infer_vector)
{
  infer_doc(doc, infer_vector);
  obj_knn_objs(NULL, infer_vector, false, true, knns, k);
}

void Doc2Vec::sent_knn_docs(TaggedDocument * doc, knn_item_t * knns, int k, real * infer_vector)
{
  infer_doc(doc, infer_vector);
  obj_knn_objs(NULL, infer_vector, false, false, knns, k);
}

real Doc2Vec::similarity(real * src, real * target)
{
  long long a;
  real dot = 0;
  for(a = 0; a < m_nn->m_dim; a++) dot += src[a] * target[a];
  return dot;
}

real Doc2Vec::distance(real * src, real * target)
{
  long long a;
  real dis = 0;
  for(a = 0; a < m_nn->m_dim; a++) dis += pow(src[a] - target[a], 2);
  return sqrt(dis);
}

void Doc2Vec::infer_doc(TaggedDocument * doc, real * vector, int skip)
{
  long long a;
  real len = 0;
  unsigned long long next_random = 1;
  for (a = 0; a < m_nn->m_dim; a++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    vector[a] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / m_nn->m_dim;
  }
  m_alpha = m_start_alpha;
  TrainModelThread trainThread(0, this, NULL, true);
  trainThread.m_doc_vector = vector;
  trainThread.buildDocument(doc, skip);
  for(a = 0; a < m_iter; a++)
  {
    trainThread.trainDocument();
    m_alpha = m_start_alpha * (1 - (a + 1.0) / m_iter);
    m_alpha = MAX(m_alpha, m_start_alpha * 0.0001);
  }
  for(a = 0; a < m_nn->m_dim; a++) len += vector[a] * vector[a];
  len = sqrt(len);
  for(a = 0; a < m_nn->m_dim; a++) vector[a] /= len;
}

real Doc2Vec::doc_likelihood(TaggedDocument * doc, int skip)
{
  if(!m_hs){
    return 0;
  }
  TrainModelThread trainThread(0, this, NULL, true);
  trainThread.buildDocument(doc, skip);
  return trainThread.doc_likelihood();
}

real Doc2Vec::context_likelihood(TaggedDocument * doc, int sentence_position)
{
  if(!m_hs){
    return 0;
  }
  if(m_word_vocab->searchVocab(doc->m_words[sentence_position]) == -1 ||
     m_word_vocab->searchVocab(doc->m_words[sentence_position]) == 0)
  {
    return 0;
  }
  TrainModelThread trainThread(0, this, NULL, true);
  trainThread.buildDocument(doc);

  long long sent_pos = sentence_position;
  for(int i = 0; i < sentence_position; i++)
  {
    long long word_idx = m_word_vocab->searchVocab(doc->m_words[i]);
    if (word_idx == -1) sent_pos--;
  }
  return trainThread.context_likelihood(sent_pos);
}

void Doc2Vec::save(FILE * fout)
{
  m_word_vocab->save(fout);
  m_doc_vocab->save(fout);
  m_nn->save(fout);
  fwrite(&m_cbow, sizeof(int), 1, fout);
  fwrite(&m_hs, sizeof(int), 1, fout);
  fwrite(&m_negtive, sizeof(int), 1, fout);
  fwrite(&m_window, sizeof(int), 1, fout);
  fwrite(&m_start_alpha, sizeof(real), 1, fout);
  fwrite(&m_sample, sizeof(real), 1, fout);
  fwrite(&m_iter, sizeof(int), 1, fout);
  m_wmd->save(fout);
}

void Doc2Vec::load(FILE * fin)
{
  m_word_vocab = new Vocabulary();
  m_word_vocab->load(fin);
  m_doc_vocab = new Vocabulary();
  m_doc_vocab->load(fin);
  m_nn = new NN();
  m_nn->load(fin);
  fread(&m_cbow, sizeof(int), 1, fin);
  fread(&m_hs, sizeof(int), 1, fin);
  fread(&m_negtive, sizeof(int), 1, fin);
  fread(&m_window, sizeof(int), 1, fin);
  fread(&m_start_alpha, sizeof(real), 1, fin);
  fread(&m_sample, sizeof(real), 1, fin);
  fread(&m_iter, sizeof(int), 1, fin);
  initNegTable();
  m_nn->norm();
  m_wmd = new WMD(this);
  m_wmd->load(fin);
}

long long Doc2Vec::dim() {return m_nn->m_dim;}
WMD * Doc2Vec::wmd() {return m_wmd;}
Vocabulary* Doc2Vec::wvocab() {return m_word_vocab;}
Vocabulary* Doc2Vec::dvocab() {return m_doc_vocab;};
NN * Doc2Vec::nn() {return m_nn;};
/////==============================DOC2VEC end========================

static void heap_adjust(knn_item_t * knns, int s, int m)
{
  real similarity = knns[s].similarity;
  long long idx = knns[s].idx;
  for(int j = 2 * s + 1; j < m; j = 2 * j + 1) {
    if(j < m - 1 && knns[j].similarity > knns[j + 1].similarity) j++;
    if(similarity < knns[j].similarity) break;
    knns[s].similarity = knns[j].similarity;
    knns[s].idx = knns[j].idx;
    s = j;
  }
  knns[s].similarity = similarity;
  knns[s].idx = idx;
}

void top_init(knn_item_t * knns, int k)
{
  for(int i = k / 2 - 1; i >= 0; i--) {
    heap_adjust(knns, i, k);
  }
}

void top_collect(knn_item_t * knns, int k, long long idx, real similarity)
{
  if(similarity <= knns[0].similarity) return;
  knns[0].similarity = similarity;
  knns[0].idx = idx;
  heap_adjust(knns, 0, k);
}

void top_sort(knn_item_t * knns, int k)
{
  real similarity;
  long long idx;
  for(int i = k - 1; i > 0; i--) {
    similarity = knns[0].similarity;
    idx = knns[0].idx;
    knns[0].similarity = knns[i].similarity;
    knns[0].idx = knns[i].idx;
    knns[i].similarity = similarity;
    knns[i].idx = idx;
    heap_adjust(knns, 0, i);
  }
}
