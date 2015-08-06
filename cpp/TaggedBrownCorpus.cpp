#include "TaggedBrownCorpus.h"

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
