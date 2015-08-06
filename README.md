# doc2vec
C++ implement of Tomas Mikolov's word/document/sentence embedding. You may want to feel the basic idea from Mikolov's two orignal papers, [word2vec](http://arxiv.org/pdf/1301.3781.pdf) and [doc2vec](http://cs.stanford.edu/~quocle/paragraph_vector.pdf)

# Dependencies
g++ </br>
[gtest 1.7+](http://code.google.com/p/googletest/) if you wanna run test suites </br>

# Why did I rewrite it in C++?
There are a few pretty nice projects like [google's word2vec](https://code.google.com/p/word2vec/) and [gensim](https://github.com/piskvorky/gensim) has already implemented the algorithm. I rewrite it for following reasons:</br>
1> speed. I believe c/c++ version has the best speed on CPU. In fact, according to test on same machine and runing with same number of theads, Fully optimized gensim verison got ~100K words/s, whereas c++ version achieved 200K words/s </br>
2> functionality. After I awared the advantage of c++ in term of efficiency, I found few open-source project 

# Getting started


