#include "common_define.h"
#include "Doc2Vec.h"

//setup parameters
char train_file[MAX_STRING], output_file[MAX_STRING];
int cbow = 1, window = 5, min_count = 1, num_threads = 4;
int hs = 1, negtive = 0;
long long dim = 100, iter = 50;
real alpha = 0.025, sample = 1e-3;

static int ArgPos(char *str, int argc, char **argv);
static void usage();
static int get_optarg(int argc, char **argv);

int ArgPos(char *str, int argc, char **argv)
{
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

//usage of main
void usage()
{
  printf("DOCUMENT/WORD VECTOR estimation toolkit\n\n");
  printf("Options:\n");
  printf("Parameters for training:\n");
  printf("\t-train <file>\n");
  printf("\t\tUse text data from <file> to train the model\n");
  printf("\t-output <file>\n");
  printf("\t\tUse <file> to save the resulting model\n");
  printf("\t-dim <int>\n");
  printf("\t\tSet dimention of document/word vectors; default is 100\n");
  printf("\t-window <int>\n");
  printf("\t\tSet max skip length between words; default is 5\n");
  printf("\t-sample <float>\n");
  printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
  printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
  printf("\t-threads <int>\n");
  printf("\t\tUse <int> threads (default 12)\n");
  printf("\t-iter <int>\n");
  printf("\t\tRun more training iterations (default 5)\n");
  printf("\t-min-count <int>\n");
  printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
  printf("\t-alpha <float>\n");
  printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
  printf("\t-cbow <int>\n");
  printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
  printf("\t-hs <int>\n");
  printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
  printf("\t-negative <int>\n");
  printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
}

//get arguments from command line
int get_optarg(int argc, char **argv)
{
  int i;
  output_file[0] = 0;
  if ((i = ArgPos((char *)"-dim", argc, argv)) > 0) dim = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negtive", argc, argv)) > 0) negtive = atoi(argv[i + 1]);
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  return output_file[0] == 0 ? -1 : 0;
}

int main(int argc, char **argv)
{
  if (argc == 1 || get_optarg(argc, argv) < 0)
  {
    usage();
    return 0;
  }
  Doc2Vec doc2vec;
  doc2vec.train(train_file, dim, cbow, hs, negtive, iter, window, alpha, sample, min_count, num_threads);
  printf("\nWrite model to %s\n", output_file);
  FILE * fout = fopen(output_file, "wb");
  doc2vec.save(fout);
  fclose(fout);
  return 0;
}
