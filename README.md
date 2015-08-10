# doc2vec
C++ implement of Tomas Mikolov's word/document embedding. You may want to feel the basic idea from Mikolov's two orignal papers, [word2vec](http://arxiv.org/pdf/1301.3781.pdf) and [doc2vec](http://cs.stanford.edu/~quocle/paragraph_vector.pdf). More recently, Andrew M. Dai etc from Google reported its power in more [detail](http://arxiv.org/pdf/1507.07998.pdf)

## Dependencies
* g++ </br>
* [gtest 1.7+](http://code.google.com/p/googletest/) if you wanna run test suites </br>

## Why did I rewrite it in C++?
There are a few pretty nice projects like [google's word2vec](https://code.google.com/p/word2vec/) and [gensim](https://github.com/piskvorky/gensim) has already implemented the algorithm, from which I learned quite a lot. However, I rewrite it for following reasons:</br>

* speed. I believe c/c++ version has the best speed on CPU. In fact, according to test on same machine and runing with same setting, Fully optimized gensim verison got ~100K words/s, whereas c++ version achieved 200K words/s </br>

* functionality. After I awared the advantage of c/c++ in term of efficiency, I found few c/c++ project implements both word and document embedding. Moreover, some important application for these embedding have not been fully developed, such as online infer document, [likelihood of document](http://arxiv.org/abs/1504.07295) and keyword extraction </br>

* scalability. I found that it's extremely slow when doing task like "most similar" on large data. One straight-forward way is distributing, the other is putting on GPUs. For these purposes, I prefer to design data structrue by myself

## Getting started
I lauched an expriment on 4,603,266 chinese academic papers' title, which has ~8 words on average. </br>
Following will show you some of the result and how-to </br>

prepare trainning file in format like this: </br>

    _*23134 Distributed Representations of Sentences and Documents
    _*31356 Document Classification by Inversion of Distributed Language Representations
    _*31345 thanks to deep learning, bring us to a new sight
    ...

train model: </br>

    Doc2Vec doc2vec;
    doc2vec.train("path-to-taining-file", 100, 0, 1, 0, 50, 5, 0.025, 1e-3, 5, 6);

save model if you want: </br>

    FILE * fout = fopen("path-to-model", "wb");
    doc2vec.save(fout);
    fclose(fout);

load model from file: </br>

    FILE * fin = fopen("path-to-model", "rb");
    doc2vec.load(fin);
    fclose(fin);

similar words: </br>
    knn_item_t knn_items[10];
    doc2vec.word_knn_words("机器学习", knn_items, 10);
    ==============机器学习===============
    分类器 -> 0.804423
    入侵检测 -> 0.791980
    决策树 -> 0.768540
    人脸 -> 0.768381
    上下文 -> 0.767411
    抽取 -> 0.764207
    语音识别 -> 0.761335
    手写 -> 0.756788
    ontology -> 0.756637
    检索 -> 0.753305

similar documents: </br>
    doc2vec.doc_knn_docs("_*1000045631_图书馆信息服务评价指标体系的构建", knn_items, 10);
    ==============_*1000045631_图书馆信息服务评价指标体系的构建===============
    _*34860843_图书馆信息服务绩效评价指标体系研究 -> 0.807343
    _*22682419_图书馆网站评价指标体系研究 -> 0.788873
    _*15508242_图书馆网站评价指标体系研究 -> 0.785874
    _*15800806_图书馆网站评价指标体系研究 -> 0.781245
    _*11365926_图书馆与社区信息服务 -> 0.779269
    _*3555700_图书馆与信息服务 -> 0.769923
    _*1000973543_构建和谐社会与图书馆服务 -> 0.751963
    _*24913702_图书馆可持续发展评价指标体系构建研究 -> 0.745885
    _*1004045506_图书馆信息服务营销 -> 0.741991
    _*38771655_图书馆2.0服务模式构建 -> 0.738804
