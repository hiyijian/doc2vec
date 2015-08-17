#include <limits>
#include <stdarg.h>
#include "gtest/gtest.h"
#include "Doc2Vec.h"
#include "WMD.h"
#include "TaggedBrownCorpus.h"
#include "common_define.h"


#define K 10
static void buildDoc(TaggedDocument * doc, ...);

class TestSimilar: public ::testing::Test{
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
  if(doc2vec.word_knn_words("遥感信息", knn_items, K)){
    print_knns("遥感信息");
  }
}

TEST_F(TestSimilar, doc_to_doc) {
  if(doc2vec.doc_knn_docs("_*1000031519_体育教学中语言艺术的探讨", knn_items, K)){
    print_knns("_*1000031519_体育教学中语言艺术的探讨");
  }
  if(doc2vec.doc_knn_docs("_*1000045631_图书馆信息服务评价指标体系的构建", knn_items, K)){
    print_knns("_*1000045631_图书馆信息服务评价指标体系的构建");
  }
  if(doc2vec.doc_knn_docs("_*1000037612_一种有效的通信电台综合识别技术", knn_items, K)){
    print_knns("_*1000037612_一种有效的通信电台综合识别技术");
  }
}

TEST_F(TestSimilar, sent_to_doc) {
  real * infer_vector = NULL;
  posix_memalign((void **)&infer_vector, 128, doc2vec.dim() * sizeof(real));

  doc.m_word_num = 6;
  buildDoc(&doc, "反求工程", "cad", "建模", "技术", "研究", "</s>");
  doc2vec.sent_knn_docs(&doc, knn_items, K, infer_vector);
  print_knns("反求工程CAD建模技术研究");

  doc.m_word_num = 5;
  buildDoc(&doc, "遥感信息", "发展战略", "与", "对策", "</s>");
  doc2vec.sent_knn_docs(&doc, knn_items, K, infer_vector);
  print_knns("遥感信息发展战略与对策");

  doc.m_word_num = 11;
  buildDoc(&doc, "光伏", "并网发电", "系统", "中",	"逆变器", "的", "设计",	"与", "控制", "方法", "</s>");
  doc2vec.sent_knn_docs(&doc, knn_items, K, infer_vector);
  print_knns("光伏并网发电系统中逆变器的设计与控制方法");

  doc.m_word_num = 7;
  buildDoc(&doc, "遥感信息", "水文", "动态", "模拟", "中", "应用", "</s>");
  doc2vec.sent_knn_docs(&doc, knn_items, K, infer_vector);
  print_knns("遥感信息水文动态模拟中应用");

  doc.m_word_num = 5;
  buildDoc(&doc, "新生儿", "败血症", "诊疗", "方案", "</s>");
  doc2vec.sent_knn_docs(&doc, knn_items, K, infer_vector);
  print_knns("新生儿败血症诊疗方案");

  free(infer_vector);
}

TEST_F(TestSimilar, wmd) {
  doc.m_word_num = 5;
  buildDoc(&doc, "遥感信息", "发展战略", "与", "对策", "</s>");
  doc2vec.wmd()->sent_knn_docs_ex(&doc, knn_items, K);
  print_knns("遥感信息发展战略与对策");

  doc.m_word_num = 7;
  buildDoc(&doc, "遥感信息", "水文", "动态", "模拟", "中", "应用", "</s>");
  doc2vec.wmd()->sent_knn_docs_ex(&doc, knn_items, K);
  print_knns("遥感信息水文动态模拟中应用");

  doc.m_word_num = 6;
  buildDoc(&doc, "反求工程", "cad", "建模", "技术", "研究", "</s>");
  doc2vec.wmd()->sent_knn_docs_ex(&doc, knn_items, K);
  print_knns("反求工程CAD建模技术研究");
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
