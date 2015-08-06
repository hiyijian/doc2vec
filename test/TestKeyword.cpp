#include <limits>
#include <stdarg.h>
#include "gtest/gtest.h"
#include "Doc2Vec.h"
#include "common_define.h"

static int compare(const void *a,const void *b);
static void buildDoc(TaggedDocument * doc, ...);

class TestKeyword: public ::testing::Test{
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
  //leave one out
  static void getLOOKeyords(TaggedDocument * doc, knn_item_t * knn_items)
  {
    real * infer_vector1 = NULL;
    posix_memalign((void **)&infer_vector1, 128, (long long)100 * sizeof(real));
    real * infer_vector2 = NULL;
    posix_memalign((void **)&infer_vector2, 128, (long long)100 * sizeof(real));
    doc2vec.infer_doc(doc, infer_vector1);

    TaggedDocument doc1;
    for(int i = 0; i < doc->m_word_num - 1; i++) {
      doc1.m_word_num = 0;
      for(int j = 0; j < doc->m_word_num; j++) if(i != j){
        strcpy(doc1.m_words[doc1.m_word_num], doc->m_words[j]);
        doc1.m_word_num ++;
      }
      doc2vec.infer_doc(&doc1, infer_vector2);
      strcpy(knn_items[i].word, doc->m_words[i]);
      knn_items[i].similarity = doc2vec.similarity(infer_vector1, infer_vector2);
    }
    qsort((void *)knn_items, doc->m_word_num - 1, sizeof(knn_item_t), compare);
    free(infer_vector1);
    free(infer_vector2);
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

  doc.m_word_num = 5;
  buildDoc(&doc, "遥感信息", "发展战略", "与", "对策", "</s>");
  knn_items = new knn_item_t[doc.m_word_num - 1];
  getLOOKeyords(&doc, knn_items);
  print_keywords("遥感信息发展战略与对策", knn_items, doc.m_word_num - 1);
  delete [] knn_items;

  doc.m_word_num = 11;
  buildDoc(&doc, "光伏", "并网发电", "系统", "中",	"逆变器", "的", "设计",	"与", "控制", "方法", "</s>");
  knn_items = new knn_item_t[doc.m_word_num - 1];
  getLOOKeyords(&doc, knn_items);
  print_keywords("光伏并网发电系统中逆变器的设计与控制方法", knn_items, doc.m_word_num - 1);
  delete [] knn_items;

  doc.m_word_num = 19;
  buildDoc(&doc, "理查", "•", "施特劳斯", "艺术", "歌曲", "的", "奏鸣", "思维", "与", "歌曲", "内容", "的", "契合", "以", "玫瑰色", "丝带", "为", "例", "</s>");
  knn_items = new knn_item_t[doc.m_word_num - 1];
  getLOOKeyords(&doc, knn_items);
  print_keywords("理查•施特劳斯艺术歌曲的奏鸣思维与歌曲内容的契合——以《攻瑰色丝带》为例", knn_items, doc.m_word_num - 1);
  delete [] knn_items;

  doc.m_word_num = 8;
  buildDoc(&doc, "理查", "施特劳斯", "歌曲", "钢琴伴奏", "和声", "艺术", "特征", "</s>");
  knn_items = new knn_item_t[doc.m_word_num - 1];
  getLOOKeyords(&doc, knn_items);
  print_keywords("理查施特劳斯歌曲钢琴伴奏和声艺术特征", knn_items, doc.m_word_num - 1);
  delete [] knn_items;

  doc.m_word_num = 12;
  buildDoc(&doc, "理查", "施特劳斯", "与", "霍夫曼", "斯塔尔", "合作关系", "研究", "以", "玫瑰骑士", "为", "例", "</s>");
  knn_items = new knn_item_t[doc.m_word_num - 1];
  getLOOKeyords(&doc, knn_items);
  print_keywords("理查施特劳斯歌曲钢琴伴奏和声艺术特征", knn_items, doc.m_word_num - 1);
  delete [] knn_items;
}
