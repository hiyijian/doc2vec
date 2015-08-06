#include "Doc2Vec.h"

static void * trainModelThread(void * params);

void * trainModelThread(void * params)
{
  TrainModelThread * tparams = (TrainModelThread *)params;
  tparams->train();
  return NULL;
}

/////==============================DOC2VEC========================
Doc2Vec::Doc2Vec(): m_word_vocab(NULL), m_doc_vocab(NULL), m_nn(NULL),
  m_expTable(NULL), m_negtive_sample_table(NULL)
{
  initExpTable();
}

Doc2Vec::~Doc2Vec()
{
  if(m_word_vocab) delete m_word_vocab;
  if(m_doc_vocab) delete m_doc_vocab;
  if(m_nn) delete m_nn;
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

  m_word_vocab = new Vocabulary(train_file, min_count);
  m_doc_vocab = new Vocabulary(train_file, 1, true);
  m_nn = new NN(m_word_vocab->m_vocab_size, m_doc_vocab->m_vocab_size, dim);
  if(m_negtive > 0) initNegTable();

  m_alpha = alpha;
  m_word_count_actual = 0;
  initTrainModelThreads(train_file, threads, iter);

  printf("Train with %d threads\n", (int)m_trainModelThreads.size());
  pthread_t *pt = (pthread_t *)malloc(m_trainModelThreads.size() * sizeof(pthread_t));
  for(size_t a = 0; a < m_trainModelThreads.size(); a++) {
    pthread_create(&pt[a], NULL, trainModelThread, (void *)m_trainModelThreads[a]);
  }
  for (size_t a = 0; a < m_trainModelThreads.size(); a++) pthread_join(pt[a], NULL);
  m_nn->norm();
}

void Doc2Vec::initTrainModelThreads(const char * train_file, int threads, int iter)
{
  long long limit = m_doc_vocab->m_vocab_size / threads;
  long long sub_size = 0;
  long long tell = 0;
  TaggedBrownCorpus corpus(train_file);
  TaggedBrownCorpus * sub_c = NULL;
  TrainModelThread * model_thread = NULL;
  TaggedDocument * doc = NULL;
  while((doc = corpus.next()) != NULL)
  {
    sub_size++;
    if(sub_size >= limit)
    {
      if(m_trainModelThreads.size() == size_t(threads - 1))
      {
        sub_c = new TaggedBrownCorpus(train_file, tell, -1);
        model_thread = new TrainModelThread(m_trainModelThreads.size(), this, sub_c, iter);
        m_trainModelThreads.push_back(model_thread);
        return;
      }
      else
      {
        sub_c = new TaggedBrownCorpus(train_file, tell, sub_size);
        model_thread = new TrainModelThread(m_trainModelThreads.size(), this, sub_c, iter);
        m_trainModelThreads.push_back(model_thread);
        tell = corpus.tell();
        sub_size = 0;
      }
    }
  }
  return;
}

bool Doc2Vec::obj_knn_objs(const char * search, real * src,
  bool search_is_word, bool target_is_word,
  knn_item_t * knns, int k)
{
  long long a = -1, b, c, d, target_size;
  real * search_vectors, * target, * target_vectors, sim;
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
  for(b = 0; b < k; b++) knns[b].similarity = -1;
  for(b = 0; b < target_size; b++)
  {
    if(search_is_word == target_is_word && a == b) continue;
    target = &(target_vectors[b * m_nn->m_dim]);
    sim = similarity(src, target);
    for (c = 0; c < k; c++) if (sim > knns[c].similarity)
    {
      for (d = k - 1; d > c; d--)
      {
        knns[d].similarity = knns[d - 1].similarity;
        strcpy(knns[d].word, knns[d - 1].word);
      }
      knns[c].similarity = sim;
      strcpy(knns[c].word, target_vocab->m_vocab[b].word);
      break;
    }
  }
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

void Doc2Vec::infer_doc(TaggedDocument * doc, real * vector, int iter)
{
  long long a;
  real len = 0;
  unsigned long long next_random = 1;
  for (a = 0; a < m_nn->m_dim; a++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    vector[a] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / m_nn->m_dim;
  }
  m_alpha = m_start_alpha;
  TrainModelThread trainThread(0, this, NULL, iter, true);
  trainThread.m_doc_vector = vector;
  trainThread.buildDocument(doc);
  for(a = 0; a < iter; a++)
  {
    trainThread.trainDocument();
    m_alpha = m_start_alpha * (1 - (a + 1.0) / iter);
    m_alpha = MAX(m_alpha, m_start_alpha * 0.0001);
  }
  for(a = 0; a < m_nn->m_dim; a++) len += vector[a] * vector[a];
  len = sqrt(len);
  for(a = 0; a < m_nn->m_dim; a++) vector[a] /= len;
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
  initNegTable();
  m_nn->norm();
}

/////==============================DOC2VEC end========================


/////==============================trainModelThread========================

TrainModelThread::TrainModelThread(long long id, Doc2Vec * doc2vec,
  TaggedBrownCorpus* sub_corpus, int iter, bool infer)
{
  m_id = id;
  m_doc2vec = doc2vec;
  m_corpus = sub_corpus;
  m_iter = iter;
  m_infer = infer;

  m_start = clock();
  m_next_random = id;
  m_sentence_length = 0;
  m_sentence_nosample_length = 0;
  m_word_count = 0;
  m_last_word_count = 0;

  m_neu1 = (real *)calloc(doc2vec->m_nn->m_dim, sizeof(real));
  m_neu1e = (real *)calloc(doc2vec->m_nn->m_dim, sizeof(real));
}

TrainModelThread::~TrainModelThread()
{
  free(m_neu1);
  free(m_neu1e);
  delete m_corpus;
}

void TrainModelThread::train()
{
  TaggedDocument * doc = NULL;
  for(long long local_iter = 0; local_iter < m_iter; local_iter++)
  {
    while((doc = m_corpus->next()) != NULL)
    {
      updateLR();
      buildDocument(doc);
      if(!m_doc_vector) continue;
      trainDocument();
    }
    m_corpus->rewind();
    m_doc2vec->m_word_count_actual += m_word_count - m_last_word_count;
    m_word_count = 0;
    m_last_word_count = 0;
  }
}

void TrainModelThread::updateLR()
{
  long long train_words = m_doc2vec->m_word_vocab->m_train_words;
  if (m_word_count - m_last_word_count > 10000) { //statistics speed per 10000 words, and update learning rate
    m_doc2vec->m_word_count_actual += m_word_count - m_last_word_count;
    m_last_word_count = m_word_count;
    clock_t now = clock();
    printf("%cAlpha: %f  Progress: %.2f%%  Words/sec: %.2fk  ", 13, m_doc2vec->m_alpha,
     m_doc2vec->m_word_count_actual / (real)(m_iter * train_words + 1) * 100,
     m_doc2vec->m_word_count_actual / ((real)(now - m_start + 1) / (real)CLOCKS_PER_SEC * 1000));
    fflush(stdout);
    m_doc2vec->m_alpha = m_doc2vec->m_start_alpha * (1 - m_doc2vec->m_word_count_actual / (real)(m_iter * train_words + 1));
    m_doc2vec->m_alpha = MAX(m_doc2vec->m_alpha, m_doc2vec->m_start_alpha * 0.0001);
  }
}

void TrainModelThread::buildDocument(TaggedDocument * doc)
{
  if(!m_infer) {
    m_doc_vector = NULL;
    long long doc_idx = m_doc2vec->m_doc_vocab->searchVocab(doc->m_tag);
    if(doc_idx < 0) {
      return;
    }
    m_doc_vector = &(m_doc2vec->m_nn->m_dsyn0[m_doc2vec->m_nn->m_dim * doc_idx]);
  }
  m_sentence_length = 0;
  m_sentence_nosample_length = 0;
  for(int i = 0; i < doc->m_word_num; i++)
  {
    long long word_idx = m_doc2vec->m_word_vocab->searchVocab(doc->m_words[i]);
    if (word_idx == -1) continue;
    if (word_idx == 0) break;
    m_word_count++;
    m_sen_nosample[m_sentence_nosample_length] = word_idx;
    m_sentence_nosample_length++;
    if(!down_sample(m_doc2vec->m_word_vocab->m_vocab[word_idx].cn))
    {
      m_sen[m_sentence_length] = word_idx;
      m_sentence_length++;
    }
  }
}

void TrainModelThread::trainSampleCbow(long long central, long long context_start, long long context_end)
{
  real f, g;
  long long a, c, d, l2, last_word, target, label, cw = 0;
  long long central_word = m_sen[central];
  long long layer1_size = m_doc2vec->m_nn->m_dim;
  real * syn0 = m_doc2vec->m_nn->m_syn0;
  real * syn1 = m_doc2vec->m_nn->m_syn1;
  real * syn1neg = m_doc2vec->m_nn->m_syn1neg;
  struct vocab_word_t* vocab = m_doc2vec->m_word_vocab->m_vocab;

  for (c = 0; c < layer1_size; c++) m_neu1[c] = 0;
  for (c = 0; c < layer1_size; c++) m_neu1e[c] = 0;
  //averge context
  for(a = context_start; a < context_end; a++) if(a != central)
  {
    last_word = m_sen[a];
    for (c = 0; c < layer1_size; c++) m_neu1[c] += syn0[c + last_word * layer1_size];
    cw++;
  }
  for (c = 0; c < layer1_size; c++) m_neu1[c] += m_doc_vector[c];
  cw++;
  for (c = 0; c < layer1_size; c++) m_neu1[c] /= cw;
  //hierarchical softmax
  if(m_doc2vec->m_hs) for (d = 0; d < vocab[central_word].codelen; d++) {
    f = 0;
    l2 = vocab[central_word].point[d] * layer1_size;
    for (c = 0; c < layer1_size; c++) f += m_neu1[c] * syn1[c + l2];
    if (f <= -MAX_EXP) continue;
    else if (f >= MAX_EXP) continue;
    else f = m_doc2vec->m_expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
    g = (1 - vocab[central_word].code[d] - f) * m_doc2vec->m_alpha;
    for (c = 0; c < layer1_size; c++) m_neu1e[c] += g * syn1[c + l2];
    if(!m_infer) for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * m_neu1[c];
  }
  //negative sampling
  if (m_doc2vec->m_negtive > 0) for (d = 0; d < m_doc2vec->m_negtive + 1; d++) {
    if (d == 0) {
      target = central_word;
      label = 1;
    } else {
      target = negtive_sample();
      if (target == central_word) continue;
      label = 0;
    }
    l2 = target * layer1_size;
    f = 0;
    for (c = 0; c < layer1_size; c++) f += m_neu1[c] * syn1neg[c + l2];
    if (f > MAX_EXP) g = (label - 1) * m_doc2vec->m_alpha;
    else if (f < -MAX_EXP) g = (label - 0) * m_doc2vec->m_alpha;
    else g = (label - m_doc2vec->m_expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * m_doc2vec->m_alpha;
    for (c = 0; c < layer1_size; c++) m_neu1e[c] += g * syn1neg[c + l2];
    if(!m_infer) for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * m_neu1[c];
  }
  if(!m_infer) for(long long a = context_start; a < context_end; a++) if(a != central)
  {
    last_word = m_sen[a];
    for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += m_neu1e[c];
  }
  for (c = 0; c < layer1_size; c++) m_doc_vector[c] += m_neu1e[c];
}

void TrainModelThread::trainPairSg(long long central_word, real * context)
{
  real f, g;
  long long c, d, l2, target, label;
  long long layer1_size = m_doc2vec->m_nn->m_dim;
  real * syn1 = m_doc2vec->m_nn->m_syn1;
  real * syn1neg = m_doc2vec->m_nn->m_syn1neg;
  for (c = 0; c < layer1_size; c++) m_neu1e[c] = 0;
  //hierarchical softmax
  if(m_doc2vec->m_hs) for (d = 0; d < m_doc2vec->m_word_vocab->m_vocab[central_word].codelen; d++)
  {
    f = 0;
    l2 = m_doc2vec->m_word_vocab->m_vocab[central_word].point[d] * layer1_size;
    for (c = 0; c < layer1_size; c++) f += context[c] * syn1[c + l2];
    if (f <= -MAX_EXP) continue;
    else if (f >= MAX_EXP) continue;
    else f = m_doc2vec->m_expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
    g = (1 - m_doc2vec->m_word_vocab->m_vocab[central_word].code[d] - f) * m_doc2vec->m_alpha;
    for (c = 0; c < layer1_size; c++) m_neu1e[c] += g * syn1[c + l2];
    if(!m_infer) for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * context[c];
  }
  //negative sampling
  if (m_doc2vec->m_negtive > 0) for (d = 0; d < m_doc2vec->m_negtive + 1; d++) {
   if (d == 0) {
     target = central_word;
     label = 1;
   } else {
     target = negtive_sample();
     if (target == central_word) continue;
     label = 0;
   }
   l2 = target * layer1_size;
   f = 0;
   for (c = 0; c < layer1_size; c++) f += context[c] * syn1neg[c + l2];
   if (f > MAX_EXP) g = (label - 1) * m_doc2vec->m_alpha;
   else if (f < -MAX_EXP) g = (label - 0) * m_doc2vec->m_alpha;
   else g = (label - m_doc2vec->m_expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * m_doc2vec->m_alpha;
   for (c = 0; c < layer1_size; c++) m_neu1e[c] += g * syn1neg[c + l2];
   if(!m_infer) for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * context[c];
  }
  for (c = 0; c < layer1_size; c++) context[c] += m_neu1e[c];
}

void TrainModelThread::trainSampleSg(long long central, long long context_start, long long context_end)
{
  long long a, last_word;
  long long central_word = m_sen[central];
  for(a = context_start; a < context_end; a++) if(a != central)
  {
    last_word = m_sen[a];
    trainPairSg(central_word, &(m_doc2vec->m_nn->m_syn0[last_word * m_doc2vec->m_nn->m_dim]));
  }
}

void TrainModelThread::trainDocument()
{
  long long sentence_position, a, b, context_start, context_end, last_word;
  for(sentence_position = 0; sentence_position < m_sentence_length; sentence_position++)
  {
    m_next_random = m_next_random * (unsigned long long)25214903917 + 11;
    b = m_next_random % m_doc2vec->m_window;
    context_start = MAX(0, sentence_position - m_doc2vec->m_window + b);
    context_end = MIN(sentence_position + m_doc2vec->m_window - b + 1, m_sentence_length);
    if(m_doc2vec->m_cbow)
    {
      trainSampleCbow(sentence_position, context_start, context_end);
    }
    else
    {
      if(!m_infer) trainSampleSg(sentence_position, context_start, context_end);
    }
  }
  if(!m_doc2vec->m_cbow)
  {
    for(a = 0; a < m_sentence_nosample_length; a++)
    {
      last_word = m_sen_nosample[a];
      trainPairSg(last_word, m_doc_vector);
    }
  }
}

bool TrainModelThread::down_sample(long long cn)
{
  if (m_doc2vec->m_sample > 0)
  {
    real ran = (sqrt(cn /
      (m_doc2vec->m_sample * m_doc2vec->m_word_vocab->m_train_words)) + 1) *
      (m_doc2vec->m_sample * m_doc2vec->m_word_vocab->m_train_words) / cn;
    m_next_random = m_next_random * (unsigned long long)25214903917 + 11;
    if (ran < (m_next_random & 0xFFFF) / (real)65536) return true;
  }
  return false;
}

long long TrainModelThread::negtive_sample()
{
  m_next_random = m_next_random * (unsigned long long)25214903917 + 11;
  long long target = m_doc2vec->m_negtive_sample_table[(m_next_random >> 16) % negtive_sample_table_size];
  if (target == 0) target = m_next_random % (m_doc2vec->m_word_vocab->m_vocab_size - 1) + 1;
  return target;
}
