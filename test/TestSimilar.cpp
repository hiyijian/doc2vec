#include <limits>
#include <stdarg.h>
#include "gtest/gtest.h"
#include "Doc2Vec.h"
#include "common_define.h"

#define K 10
static void buildDoc(TaggedDocument * doc, ...);

class TestSimilar: public ::testing::Test{
protected:
  static void SetUpTestCase() {
    FILE * fin = fopen("../data/model.sg", "rb");
    doc2vec.load(fin);
    fclose(fin);
  }
  static void TearDownTestCase() {}
  virtual void SetUp() { }
  virtual void TearDown() {}

public:
  static void print_knns(const char * search) {
    printf("==============%s===============\n", search);
    for(int a = 0; a < K; a++) {
      printf("%s -> %f\n", knn_items[a].word, knn_items[a].similarity);
    }
  }

public:
  static Doc2Vec doc2vec;
  static TaggedDocument doc;
  static knn_item_t knn_items[K];
};
Doc2Vec TestSimilar::doc2vec;
TaggedDocument TestSimilar::doc;
knn_item_t TestSimilar::knn_items[K];

TEST_F(TestSimilar, word_to_word) {
  if(doc2vec.word_knn_words("svm", knn_items, K)){
    print_knns("svm");
  }
  if(doc2vec.word_knn_words("机器学习", knn_items, K)){
    print_knns("机器学习");
  }
}

TEST_F(TestSimilar, doc_to_doc) {
  if(doc2vec.doc_knn_docs("_*9995573", knn_items, K)){
    print_knns("_*9995573");
  }
  if(doc2vec.doc_knn_docs("_*9997193", knn_items, K)){
    print_knns("_*9997193");
  }
}

TEST_F(TestSimilar, sent_to_doc) {
  real * infer_vector = NULL;
  posix_memalign((void **)&infer_vector, 128, (long long)100 * sizeof(real));

  doc.m_word_num = 6;
  buildDoc(&doc, "反求工程", "cad", "建模", "技术", "研究", "</s>");
  doc2vec.sent_knn_docs(&doc, knn_items, K, infer_vector);
  print_knns("反求工程CAD建模技术研究");

  doc.m_word_num = 5;
  buildDoc(&doc, "遥感信息", "发展战略", "与", "对策", "</s>");
  doc2vec.sent_knn_docs(&doc, knn_items, K, infer_vector);
  print_knns("遥感信息发展战略与对策");

  free(infer_vector);
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
