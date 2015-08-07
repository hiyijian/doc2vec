#include <limits>
#include "gtest/gtest.h"
#include "Doc2Vec.h"
#include "common_define.h"

TEST(TestTrain, cbow) {
  Doc2Vec doc2vec;
  doc2vec.train("../data/paper.seg", 50, 1, 1, 0, 50, 5, 0.05, 1e-3, 1, 6);
  printf("\nWrite model to %s\n", "../data/model.cbow");
  FILE * fout = fopen("../data/model.cbow", "wb");
  doc2vec.save(fout);
  fclose(fout);
}


TEST(TestTrain, sg) {
  Doc2Vec doc2vec;
  doc2vec.train("../data/paper.seg", 50, 0, 1, 0, 50, 5, 0.025, 1e-3, 1, 6);
  printf("\nWrite model to %s\n", "../data/model.sg");
  FILE * fout = fopen("../data/model.sg", "wb");
  doc2vec.save(fout);
  fclose(fout);
}
