#include "TaggedBrownCorpus.h"
#include "Vocab.h"
#include "NN.h"
#include "Doc2Vec.h"
#include <set>
#include <map>
#include <vector>

//=======================TaggedBrownCorpus=======================
TaggedBrownCorpus::TaggedBrownCorpus(const char * train_file, long long seek, long long limit_doc):
  m_seek(seek), m_doc_num(0), m_limit_doc(limit_doc)
{
  m_fin = fopen(train_file, "rb");
  if (m_fin == NULL)
  {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(m_fin, m_seek, SEEK_SET);
}

TaggedBrownCorpus::~TaggedBrownCorpus()
{
  fclose(m_fin);
}

void TaggedBrownCorpus::rewind()
{
  fseek(m_fin, m_seek, SEEK_SET);
  m_doc_num = 0;
}

TaggedDocument * TaggedBrownCorpus::next()
{
  if(feof(m_fin) || (m_limit_doc >= 0 && m_doc_num >= m_limit_doc))
  {
    return NULL;
  }
  readWord(m_doc.m_tag);
  m_doc.m_word_num = 0;
  int eol = 0;
  while(m_doc.m_word_num < MAX_SENTENCE_LENGTH && 0 == eol)
  {
    eol = readWord(m_doc.m_words[m_doc.m_word_num]);
    m_doc.m_word_num++;
  }
  m_doc_num++;
  return &m_doc;
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
// paading </s> to the EOL
//return 0 : word, return -1: EOL
int TaggedBrownCorpus::readWord(char *word)
{
  int a = 0, ch;
  while (!feof(m_fin))
  {
    ch = fgetc(m_fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n'))
    {
      if (a > 0)
      {
        if (ch == '\n') ungetc(ch, m_fin);
        break;
      }
      if (ch == '\n')
      {
        strcpy(word, (char *)"</s>");
        return -1;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
  return 0;
}

//=======================TaggedDocument=======================
TaggedDocument::TaggedDocument()
{
  m_word_num = 0;
  m_tag = (char *)calloc(MAX_STRING, sizeof(char));
  m_words = (char **)calloc(MAX_SENTENCE_LENGTH, sizeof(char*));
  for(int i = 0; i < MAX_SENTENCE_LENGTH; i++) m_words[i] = (char *)calloc(MAX_STRING, sizeof(char));
}

TaggedDocument::~TaggedDocument()
{
  free(m_tag);
  for(int i = 0; i < MAX_SENTENCE_LENGTH; i++) free(m_words[i]);
  free(m_words);
}

// //////////////UnWeightedDocument/////////////////////////////
UnWeightedDocument::UnWeightedDocument() : m_words_idx(NULL), m_word_num(0) {}

UnWeightedDocument::UnWeightedDocument(Doc2Vec * doc2vec, TaggedDocument * doc):
  m_words_idx(NULL), m_word_num(0)
{
  int a;
  long long word_idx;
  char * word;
  std::set<long long> dict;
  std::vector<long long> words_idx;
  for(a = 0; a < doc->m_word_num; a++)
  {
    word = doc->m_words[a];
    word_idx = doc2vec->m_word_vocab->searchVocab(word);
    if (word_idx == -1) continue;
    if (word_idx == 0) break;
    if(dict.find(word_idx) == dict.end()){
      dict.insert(word_idx);
      words_idx.push_back(word_idx);
    }
  }
  m_word_num = words_idx.size();
  if(m_word_num <= 0) return;
  m_words_idx = new long long[m_word_num];
  for(a = 0; a < m_word_num; a++) m_words_idx[a] = words_idx[a];
}

UnWeightedDocument::~UnWeightedDocument()
{
  if(m_words_idx) delete [] m_words_idx;
}

void UnWeightedDocument::save(FILE * fout)
{
  fwrite(&m_word_num, sizeof(int), 1, fout);
  if(m_word_num > 0) fwrite(m_words_idx, sizeof(long long), m_word_num, fout);
}
void UnWeightedDocument::load(FILE * fin)
{
  fread(&m_word_num, sizeof(int), 1, fin);
  if(m_word_num > 0)
  {
    m_words_idx = new long long[m_word_num];
    fread(m_words_idx, sizeof(long long), m_word_num, fin);
  }
  else m_words_idx = NULL;
}

// //////////////WeightedDocument/////////////////////////////
WeightedDocument::WeightedDocument(Doc2Vec * doc2vec, TaggedDocument * doc):
  UnWeightedDocument(doc2vec, doc), m_words_wei(NULL)
{
  int a;
  long long word_idx;
  char * word;
  real sim, * doc_vector = NULL, * infer_vector = NULL;
  real sum = 0;
  std::map<long long, real> scores;
  posix_memalign((void **)&doc_vector, 128, doc2vec->m_nn->m_dim * sizeof(real));
  posix_memalign((void **)&infer_vector, 128, doc2vec->m_nn->m_dim * sizeof(real));
  doc2vec->infer_doc(doc, doc_vector);
  for(a = 0; a < doc->m_word_num; a++)
  {
    word = doc->m_words[a];
    word_idx = doc2vec->m_word_vocab->searchVocab(word);
    if (word_idx == -1) continue;
    if (word_idx == 0) break;
    doc2vec->infer_doc(doc, infer_vector, a);
    sim = doc2vec->similarity(doc_vector, infer_vector);
    scores[word_idx] = pow(1.0 - sim, 1.5);
  }
  free(doc_vector);
  free(infer_vector);
  if(m_word_num <= 0) return;
  m_words_wei = new real[m_word_num];
  for(a = 0; a < m_word_num; a++) m_words_wei[a] = scores[m_words_idx[a]];
  for(a = 0; a < m_word_num; a++) sum +=  m_words_wei[a];
  for(a = 0; a < m_word_num; a++) m_words_wei[a] /= sum;
}

WeightedDocument::~WeightedDocument()
{
  if(m_words_wei) delete [] m_words_wei;
}
