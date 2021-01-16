#!/bin/sh
dir="text_clsf_lib/preprocessing/vectorization/resources/embeddings/glove/wiki"
mkdir -p $dir && \
cd $dir && \
curl http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip > glove_wiki.zip && \
unzip glove_wiki.zip && \
rm glove_wiki.zip
mv glove.6B.50d.txt 50d.txt
mv glove.6B.100d.txt 100d.txt
mv glove.6B.200d.txt 200d.txt
mv glove.6B.300d.txt 300d.txt