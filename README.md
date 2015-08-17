# doc2vec
C++ implement of Tomas Mikolov's word/document embedding. You may want to feel the basic idea from Mikolov's two orignal papers, [word2vec](http://arxiv.org/pdf/1301.3781.pdf) and [doc2vec](http://cs.stanford.edu/~quocle/paragraph_vector.pdf). More recently, Andrew M. Dai etc from Google reported its power in more [detail](http://arxiv.org/pdf/1507.07998.pdf)

## Dependencies
* g++ </br>
* [gtest 1.7+](http://code.google.com/p/googletest/) if you wanna run test suites </br>

## Why did I rewrite it in C++?
There are a few pretty nice projects like [google's word2vec](https://code.google.com/p/word2vec/) and [gensim](https://github.com/piskvorky/gensim) has already implemented the algorithm, from which I learned quite a lot. However, I rewrite it for following reasons:</br>

* speed. I believe c/c++ version has the best speed on CPU. In fact, according to test on same machine and runing with same setting, Fully optimized gensim verison got ~100K words/s, whereas c++ version achieved 200K words/s </br>

* functionality. After I awared the advantage of c/c++ in term of efficiency, I found few c/c++ project implements both word and document embedding. Moreover, some important application for these embedding have not been fully developed, such as online infer document, [likelihood of document](http://arxiv.org/abs/1504.07295), [wmd](jmlr.org/proceedings/papers/v37/kusnerb15.pdf) and keyword extraction </br>

* scalability. I found that it's extremely slow when doing task like "most similar" on large data. One straight-forward way is distributing, the other is putting on GPUs. For these purposes, I prefer to design data structrue by myself

## Getting started
I lauched an expriment on 7,987,287 chinese academic papers' title, which has ~8 words on average. </br>
Following will show you some of the result and how-to </br>

prepare trainning file in format like this: </br>

    _*23134 Distributed Representations of Sentences and Documents
    _*31356 Document Classification by Inversion of Distributed Language Representations
    _*31345 thanks to deep learning, bring us to a new sight
    ...

train model: </br>

    Doc2Vec doc2vec;
    doc2vec.train("path-to-taining-file", 50, 0, 1, 0, 15, 10, 0.025, 1e-5, 3, 6);

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
    分类器 -> 0.938366
    贝叶斯 -> 0.934191
    上下文 -> 0.931081
    推理 -> 0.927534
    向量空间 -> 0.925619
    抽取 -> 0.922363
    svm -> 0.919414
    视图 -> 0.908361
    决策树 -> 0.906520
    bayesian -> 0.904839

similar documents: </br>

    doc2vec.doc_knn_docs("_*1000045631_图书馆信息服务评价指标体系的构建", knn_items, 10);
    ==============_*1000045631_图书馆信息服务评价指标体系的构建===============
    _*43541596_公共图书馆政府信息服务绩效评价指标体系的构建 -> 0.841705
    _*32615763_图书馆电子服务质量评价指标体系构建 -> 0.833922
    _*34860843_图书馆信息服务绩效评价指标体系研究 -> 0.823280
    _*1001040838_图书馆信息资源建设评价指标体系构建研究 -> 0.822471
    _*33909320_图书馆信息服务创新研究 -> 0.814595
    _*3555700_图书馆与信息服务 -> 0.809152
    _*45080320_图书馆信息服务创新理论基础研究 -> 0.808122
    _*35544806_高校数字图书馆评价指标体系研究 -> 0.807870
    _*6427226_图书馆信息服务创新体系 -> 0.804898
    _*22765211_基于网格的图书馆信息服务 -> 0.802323

infer sentence online and get similar documents: </br>

    //a relative good case
    doc.m_word_num = 11;
    buildDoc(&doc, "光伏", "并网发电", "系统", "中",	"逆变器", "的", "设计",	"与", "控制", "方法", "</s>");
    doc2vec.sent_knn_docs(&doc, knn_items, 10, infer_vector);
    ==============光伏并网发电系统中逆变器的设计与控制方法===============
    _*23050250_光伏并网发电系统中逆变器的设计与控制方法 -> 0.923025
    _*41522718_基于级联逆变器的光伏并网发电系统控制策略 -> 0.832144
    _*7724486_光伏并网发电系统的控制方法 -> 0.787163
    _*8031480_小功率光伏并网逆变系统的研制 -> 0.783204
    _*7157773_小功率光伏并网逆变器控制的设计 -> 0.782370
    _*1001026414_光伏并网发电系统的控制策略研究 -> 0.771843
    _*33269514_光伏并网发电系统的MPPT-电压控制策略仿真 -> 0.763618
    _*37612602_光伏并网逆变器的设计与控制 -> 0.763109
    _*20065898_RTDS应用于直流控制保护系统的仿真试验 -> 0.762502
    _*41961421_光伏并网发电系统中孤岛现象的研究 -> 0.760718

    //a pretty bad case
    doc.m_word_num = 5;
    buildDoc(&doc, "遥感信息", "发展战略", "与", "对策", "</s>");
    doc2vec.sent_knn_docs(&doc, knn_items, 10, infer_vector);
    ==============遥感信息发展战略与对策===============
    _*29022751_中国水稻遥感信息获取区划研究 -> 0.717881
    _*21205970_我国观光果园的发展现状、存在问题与对策 -> 0.716743
    _*1308712_中国土地资源态势潜力及对策 -> 0.714717
    _*9456568_我国分布式能源发展战略探讨 -> 0.705569
    _*11726518_论中国能源发展战略及对策 -> 0.703126
    _*4924333_我国城市环境信息化建设发展战略探讨 -> 0.701261
    _*6419896_农业结构调整的障碍与对策分析 -> 0.698340
    _*21638304_中国企业对外直接投资障碍与对策分析 -> 0.697608
    _*4505223_中国花卉产业现状及发展战略 -> 0.697182
    _*18092739_中国能源发展战略与石油安全对策研究 -> 0.693231

somthing more interesting is that task of keyword extraction could also benefit from it(via leave-one-out, see codes in test suites): <br>

    ================ 遥感信息发展战略与对策 ===================
    遥感信息 -> -16.50
    发展战略 -> -21.81
    对策 -> -25.22
    与 -> -63.49

    ================ 光伏并网发电系统中逆变器的设计与控制方法 ===================
    逆变器 -> -167.38
    并网发电 -> -218.84
    光伏 -> -226.92
    系统 -> -284.51
    控制 -> -413.66
    设计 -> -419.63
    的 -> -535.65
    与 -> -535.65
    方法 -> -535.65
    中 -> -577.80

further more, I use these word weights to modify the [wmd](jmlr.org/proceedings/papers/v37/kusnerb15.pdf), impoving performance of document similarity:

    //the bad case methioned above turned to be acceptable
    doc.m_word_num = 6;
    buildDoc(&doc, "遥感信息", "发展战略", "与", "对策", "</s>");
    doc2vec.wmd()->sent_knn_docs_ex(&doc, knn_items, K);
    ==============遥感信息发展战略与对策===============
    _*6448414_信息时代的经济发展与遥感信息技术集成 -> -0.100316
    _*10541315_林业资源遥感信息的尺度问题研究 -> -0.130996
    _*40985040_青藏高原湖泊遥感信息提取及湖面动态变化趋势研究 -> -0.139246
    _*29022751_中国水稻遥感信息获取区划研究 -> -0.146397
    _*38794473_陕西省土地利用／覆盖变化以及驱动机制分析——基于遥感信息与 -> -0.150292
    _*9827563_基于遥感信息的土地资源可持续利用研究 -> -0.163499
    _*43973701_基于遥感信息与水稻模型相结合对镇江地区水稻种植面积与产量的 -> -0.171528
    _*10487843_济南南部地区城市扩展遥感信息动态分析 -> -0.174100
    _*8443457_论土地利用与覆盖变化遥感信息提取技术框架 -> -0.176658
