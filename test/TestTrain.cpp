#include <limits>
#include "gtest/gtest.h"
#include "Doc2Vec.h"
#include "common_define.h"

TEST(TestTrain, title_sg) {
  Doc2Vec doc2vec;
  doc2vec.train("../data/paper.title.seg", 100, 0, 1, 0, 50, 5, 0.025, 1e-3, 5, 6);
  printf("\nWrite model to %s\n", "../data/model.title.sg");
  FILE * fout = fopen("../data/model.title.sg", "wb");
  doc2vec.save(fout);
  fclose(fout);
}
