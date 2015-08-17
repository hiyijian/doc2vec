#include <limits>
#include <stdarg.h>
#include "gtest/gtest.h"
#include "Doc2Vec.h"
#include "TaggedBrownCorpus.h"
#include "common_define.h"
#include "NN.h"
#include "Vocab.h"

static int compare(const void *a,const void *b);
static void buildDoc(TaggedDocument * doc, ...);

class TestKeyword: public ::testing::Test{
protected:
  static void SetUpTestCase() {
    FILE * fin = fopen("../data/model.title.sg", "rb");
    doc2vec.load(fin);
    fclose(fin);
  }
  static void TearDownTestCase() {}
  virtual void SetUp() { }
  virtual void TearDown() {}

public:
  //leave one out
  static void getLOOKeyords(TaggedDocument * doc, knn_item_t * knn_items)
  {
    real * infer_vector1 = NULL;
    posix_memalign((void **)&infer_vector1, 128, doc2vec.dim() * sizeof(real));
    real * infer_vector2 = NULL;
    posix_memalign((void **)&infer_vector2, 128, doc2vec.dim() * sizeof(real));
    doc2vec.infer_doc(doc, infer_vector1);

    for(int i = 0; i < doc->m_word_num - 1; i++) {
      doc2vec.infer_doc(doc, infer_vector2, i);
      strcpy(knn_items[i].word, doc->m_words[i]);
      knn_items[i].similarity = doc2vec.similarity(infer_vector1, infer_vector2);
    }
    qsort((void *)knn_items, doc->m_word_num - 1, sizeof(knn_item_t), compare);
    free(infer_vector1);
    free(infer_vector2);
  }

  //similar
  static void getSimKeywords(TaggedDocument * doc, knn_item_t * knn_items)
  {
    long long word_idx;
    real * infer_vector = NULL, *wv = NULL;
    posix_memalign((void **)&infer_vector, 128, doc2vec.dim() * sizeof(real));
    doc2vec.infer_doc(doc, infer_vector);
    for(int i = 0; i < doc->m_word_num - 1; i++) {
      strcpy(knn_items[i].word, doc->m_words[i]);
      word_idx = doc2vec.wvocab()->searchVocab(doc->m_words[i]);
      if(word_idx <= 0) continue;
      wv = &(doc2vec.nn()->m_syn0norm[word_idx * doc2vec.dim()]);
      knn_items[i].similarity = doc2vec.similarity(infer_vector, wv);
    }
    qsort((void *)knn_items, doc->m_word_num - 1, sizeof(knn_item_t), compare);
    free(infer_vector);
  }

  //likelihood
  static void getLKHKeyords(TaggedDocument * doc, knn_item_t * knn_items)
  {
    TaggedDocument doc1;
    for(int i = 0; i < doc->m_word_num - 1; i++) {
      strcpy(knn_items[i].word, doc->m_words[i]);
      knn_items[i].similarity = doc2vec.doc_likelihood(doc, i);
    }
    qsort((void *)knn_items, doc->m_word_num - 1, sizeof(knn_item_t), compare);
  }

  static void print_keywords(const char* word, knn_item_t * knn_items, int n)
  {
    printf("================ %s ===================\n", word);
    for(int a = 0; a < n; a++)
    {
      printf("%s -> %.2f\n", knn_items[a].word, knn_items[a].similarity);
    }
  }

public:
  static Doc2Vec doc2vec;
};
Doc2Vec TestKeyword::doc2vec;

int compare(const void *a,const void *b)
{
  knn_item_t * aa = (knn_item_t*)a;
  knn_item_t * bb = (knn_item_t*)b;
  if(bb->similarity - aa->similarity > 0)
    return -1;
  else if(bb->similarity - aa->similarity < 0)
    return 1;
  else
    return 0;
}

void buildDoc(TaggedDocument * doc, ...)
{
  va_list pArg;
  va_start(pArg, doc);
  for(int i = 0; i < doc->m_word_num; i++){
    strcpy(doc->m_words[i], va_arg(pArg, char*));
  }
  va_end(pArg);
}

TEST_F(TestKeyword, LOO) {
  TaggedDocument doc;
  knn_item_t * knn_items = NULL;

  doc.m_word_num = 6;
  buildDoc(&doc, "反求工程", "cad", "建模", "技术", "研究", "</s>");
  knn_items = new knn_item_t[doc.m_word_num - 1];
  getLOOKeyords(&doc, knn_items);
  print_keywords("反求工程CAD建模技术研究", knn_items, doc.m_word_num - 1);
  delete [] knn_items;

  doc.m_word_num = 5;
  buildDoc(&doc, "遥感信息", "发展战略", "与", "对策", "</s>");
  knn_items = new knn_item_t[doc.m_word_num - 1];
  getLOOKeyords(&doc, knn_items);
  print_keywords("遥感信息发展战略与对策", knn_items, doc.m_word_num - 1);
  delete [] knn_items;

  doc.m_word_num = 7;
  buildDoc(&doc, "遥感信息", "水文", "动态", "模拟", "中", "应用", "</s>");
  knn_items = new knn_item_t[doc.m_word_num - 1];
  getLOOKeyords(&doc, knn_items);
  print_keywords("遥感信息水文动态模拟中应用", knn_items, doc.m_word_num - 1);
  delete [] knn_items;

  doc.m_word_num = 11;
  buildDoc(&doc, "光伏", "并网发电", "系统", "中",	"逆变器", "的", "设计",	"与", "控制", "方法", "</s>");
  knn_items = new knn_item_t[doc.m_word_num - 1];
  getLOOKeyords(&doc, knn_items);
  print_keywords("光伏并网发电系统中逆变器的设计与控制方法", knn_items, doc.m_word_num - 1);
  delete [] knn_items;
}

TEST_F(TestKeyword, LKH) {
  TaggedDocument doc;
  knn_item_t * knn_items = NULL;

  doc.m_word_num = 5;
  buildDoc(&doc, "遥感信息", "发展战略", "与", "对策", "</s>");
  knn_items = new knn_item_t[doc.m_word_num - 1];
  getLKHKeyords(&doc, knn_items);
  print_keywords("遥感信息发展战略与对策", knn_items, doc.m_word_num - 1);
  delete [] knn_items;

  doc.m_word_num = 7;
  buildDoc(&doc, "遥感信息", "水文", "动态", "模拟", "中", "应用", "</s>");
  knn_items = new knn_item_t[doc.m_word_num - 1];
  getLKHKeyords(&doc, knn_items);
  print_keywords("遥感信息水文动态模拟中应用", knn_items, doc.m_word_num - 1);
  delete [] knn_items;


  doc.m_word_num = 11;
  buildDoc(&doc, "光伏", "并网发电", "系统", "中",	"逆变器", "的", "设计",	"与", "控制", "方法", "</s>");
  knn_items = new knn_item_t[doc.m_word_num - 1];
  getLKHKeyords(&doc, knn_items);
  print_keywords("光伏并网发电系统中逆变器的设计与控制方法", knn_items, doc.m_word_num - 1);
  delete [] knn_items;
}


TEST_F(TestKeyword, SIM) {
  TaggedDocument doc;
  knn_item_t * knn_items = NULL;

  doc.m_word_num = 5;
  buildDoc(&doc, "遥感信息", "发展战略", "与", "对策", "</s>");
  knn_items = new knn_item_t[doc.m_word_num - 1];
  getSimKeywords(&doc, knn_items);
  print_keywords("遥感信息发展战略与对策", knn_items, doc.m_word_num - 1);
  delete [] knn_items;

  doc.m_word_num = 7;
  buildDoc(&doc, "遥感信息", "水文", "动态", "模拟", "中", "应用", "</s>");
  knn_items = new knn_item_t[doc.m_word_num - 1];
  getSimKeywords(&doc, knn_items);
  print_keywords("遥感信息水文动态模拟中应用", knn_items, doc.m_word_num - 1);
  delete [] knn_items;


  doc.m_word_num = 11;
  buildDoc(&doc, "光伏", "并网发电", "系统", "中",	"逆变器", "的", "设计",	"与", "控制", "方法", "</s>");
  knn_items = new knn_item_t[doc.m_word_num - 1];
  getSimKeywords(&doc, knn_items);
  print_keywords("光伏并网发电系统中逆变器的设计与控制方法", knn_items, doc.m_word_num - 1);
  delete [] knn_items;
}
