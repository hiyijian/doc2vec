#include "Vocab.h"
#include "TaggedBrownCorpus.h"

static int vocabCompare(const void *a, const void *b);
static int getWordHash(const char *word);

Vocabulary::Vocabulary(const char * train_file, int min_count, bool doctag) :
  m_vocab(NULL), m_vocab_size(0), m_train_words(0), m_vocab_capacity(1000),
  m_vocab_hash(NULL), m_min_reduce(1), m_min_count(min_count), m_doctag(doctag)
{
  if(m_doctag) m_min_count = 1;
  m_vocab = (struct vocab_word_t *)calloc(m_vocab_capacity, sizeof(struct vocab_word_t));
  m_vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  loadFromTrainFile(train_file);
  if(!m_doctag) createHuffmanTree();
}

Vocabulary::~Vocabulary()
{
  free(m_vocab);
  free(m_vocab_hash);
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
long long Vocabulary::searchVocab(const char *word)
{
  unsigned int hash = getWordHash(word);
  while (1)
  {
    if (m_vocab_hash[hash] == -1 || m_vocab[m_vocab_hash[hash]].word == NULL) return -1;
    if (!strcmp(word, m_vocab[m_vocab_hash[hash]].word)) return m_vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

void Vocabulary::loadFromTrainFile(const char * train_file)
{
  char * word;
  TaggedBrownCorpus corpus(train_file);
  long long a, i, k;
  for (a = 0; a < vocab_hash_size; a++) m_vocab_hash[a] = -1;
  m_vocab_size = 0;
  if(!m_doctag) addWordToVocab((char *)"</s>");
  TaggedDocument * doc = NULL;
  while ((doc = corpus.next()) != NULL) {
    if(m_doctag) {  //for doc tag
      word = doc->m_tag;
      m_train_words++;
      i = searchVocab(word);
      if (i == -1) {
        a = addWordToVocab(word);
        m_vocab[a].cn = 1;
      }
    } else { // for doc words
      for(k = 0; k < doc->m_word_num; k++){
        word = doc->m_words[k];
        m_train_words++;
        if (!m_doctag && m_train_words % 100000 == 0)
        {
          printf("%lldK%c", m_train_words / 1000, 13);
          fflush(stdout);
        }
        i = searchVocab(word);
        if (i == -1) {
          a = addWordToVocab(word);
          m_vocab[a].cn = 1;
        } else m_vocab[i].cn++;
        if (m_vocab_size > vocab_hash_size * 0.7) reduceVocab();
      }
      m_train_words--;
    }
  }
  if(!m_doctag)
  {
    sortVocab();
    printf("Vocab size: %lld\n", m_vocab_size);
    printf("Words in train file: %lld\n", m_train_words);
  }
}

long long Vocabulary::addWordToVocab(const char *word)
{
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  m_vocab[m_vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(m_vocab[m_vocab_size].word, word);
  m_vocab[m_vocab_size].cn = 0;
  m_vocab_size++;
  // Reallocate memory if needed
  if (m_vocab_size + 2 >= m_vocab_capacity)
  {
    m_vocab_capacity += 1000;
    m_vocab = (struct vocab_word_t *)realloc(m_vocab, m_vocab_capacity * sizeof(struct vocab_word_t));
  }
  hash = getWordHash(word);
  while (m_vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  m_vocab_hash[hash] = m_vocab_size - 1;
  return m_vocab_size - 1;
}

// Sorts the vocabulary by frequency using word counts, frequent->infrequent
void Vocabulary::sortVocab()
{
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&m_vocab[1], m_vocab_size - 1, sizeof(struct vocab_word_t), vocabCompare);
  //reduce words and re-hash
  for (a = 0; a < vocab_hash_size; a++) m_vocab_hash[a] = -1;
  size = m_vocab_size;
  m_train_words = 0;
  for (a = 0; a < size; a++)
  {
    // Words occuring less than min_count times will be discarded from the vocab
    if (m_vocab[a].cn < m_min_count)
    {
      m_vocab_size--;
      free(m_vocab[m_vocab_size].word);
      m_vocab[m_vocab_size].word = NULL;
    }
    else
    {
      // Hash will be re-computed, as after the sorting it is not actual
      hash = getWordHash(m_vocab[a].word);
      while (m_vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      m_vocab_hash[hash] = a;
      m_train_words += m_vocab[a].cn;
    }
  }
  m_train_words -= m_vocab[0].cn; //exclude <s>
  m_vocab = (struct vocab_word_t *)realloc(m_vocab, (m_vocab_size + 1) * sizeof(struct vocab_word_t));
}

// Reduces the vocabulary by removing infrequent tokens
void Vocabulary::reduceVocab()
{
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < m_vocab_size; a++) if (m_vocab[a].cn > m_min_reduce)
  {
    m_vocab[b].cn = m_vocab[a].cn;
    m_vocab[b].word = m_vocab[a].word;
    b++;
  } else free(m_vocab[a].word);
  m_vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) m_vocab_hash[a] = -1;
  for (a = 0; a < m_vocab_size; a++)
  {
    // Hash will be re-computed, as it is not actual
    hash = getWordHash(m_vocab[a].word);
    while (m_vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    m_vocab_hash[hash] = a;
  }
  fflush(stdout);
  m_min_reduce++;
}

void Vocabulary::createHuffmanTree()
{
  // Allocate memory for the binary tree construction
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(m_vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(m_vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(m_vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < m_vocab_size; a++) {
    m_vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    m_vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
  for (a = 0; a < m_vocab_size; a++) count[a] = m_vocab[a].cn;
  for (a = m_vocab_size; a < m_vocab_size * 2; a++) count[a] = 1e15;
  pos1 = m_vocab_size - 1;
  pos2 = m_vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < m_vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[m_vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = m_vocab_size + a;
    parent_node[min2i] = m_vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < m_vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == m_vocab_size * 2 - 2) break;
    }
    m_vocab[a].codelen = i;
    m_vocab[a].point[0] = m_vocab_size - 2;
    for (b = 0; b < i; b++) {
      m_vocab[a].code[i - b - 1] = code[b];
      m_vocab[a].point[i - b] = point[b] - m_vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void Vocabulary::save(FILE * fout)
{
  long long a;
  int wordlen;
  fwrite(&m_vocab_size, sizeof(long long), 1, fout);
  fwrite(&m_train_words, sizeof(long long), 1, fout);
  fwrite(&m_vocab_capacity, sizeof(long long), 1, fout);
  fwrite(&m_min_reduce, sizeof(int), 1, fout);
  fwrite(&m_min_count, sizeof(int), 1, fout);
  fwrite(&m_doctag, sizeof(bool), 1, fout);
  for(a = 0; a < m_vocab_size; a++)
  {
    wordlen = strlen(m_vocab[a].word);
    fwrite(&wordlen, sizeof(int), 1, fout);
    fwrite(m_vocab[a].word, sizeof(char), wordlen, fout);
    fwrite(&(m_vocab[a].cn), sizeof(long long), 1, fout);
    if(!m_doctag)
    {
      fwrite(&(m_vocab[a].codelen), sizeof(char), 1, fout);
      fwrite(m_vocab[a].point, sizeof(int), m_vocab[a].codelen, fout);
      fwrite(m_vocab[a].code, sizeof(char), m_vocab[a].codelen, fout);
    }
  }
  fwrite(m_vocab_hash, sizeof(int), vocab_hash_size, fout);
}

void Vocabulary::load(FILE * fin)
{
  long long a;
  int wordlen;
  fread(&m_vocab_size, sizeof(long long), 1, fin);
  fread(&m_train_words, sizeof(long long), 1, fin);
  fread(&m_vocab_capacity, sizeof(long long), 1, fin);
  fread(&m_min_reduce, sizeof(int), 1, fin);
  fread(&m_min_count, sizeof(int), 1, fin);
  fread(&m_doctag, sizeof(bool), 1, fin);
  m_vocab = (struct vocab_word_t *)calloc(m_vocab_capacity, sizeof(struct vocab_word_t));
  for(a = 0; a < m_vocab_size; a++)
  {
    fread(&wordlen, sizeof(int), 1, fin);
    m_vocab[a].word = (char *)calloc(wordlen + 1, sizeof(char));
    fread(m_vocab[a].word, sizeof(char), wordlen, fin);
    fread(&(m_vocab[a].cn), sizeof(long long), 1, fin);
    if(!m_doctag)
    {
      fread(&(m_vocab[a].codelen), sizeof(char), 1, fin);
      m_vocab[a].point = (int *)calloc(m_vocab[a].codelen, sizeof(int));
      fread(m_vocab[a].point, sizeof(int), m_vocab[a].codelen, fin);
      m_vocab[a].code = (char *)calloc(m_vocab[a].codelen, sizeof(char));
      fread(m_vocab[a].code, sizeof(char), m_vocab[a].codelen, fin);
    }
  }
  m_vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  fread(m_vocab_hash, sizeof(int), vocab_hash_size, fin);
}

int vocabCompare(const void *a, const void *b)
{
    return ((struct vocab_word_t *)b)->cn - ((struct vocab_word_t *)a)->cn;
}

// Returns hash value of a word
int getWordHash(const char *word)
{
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}
