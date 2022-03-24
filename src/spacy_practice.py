#!/usr/bin/env python
# -*- coding: utf-8 -*-

######################
#title:spaCy勉強用
#author:Koya Okuse
#date:2022/03/23
######################

import spacy

#英語のトークナイザー、タガー、parser、NER、word vectorをインポート
nlp = spacy.load('en_core_web_sm')
print("------------------------------------------------------")
#サンプルテキストに対する固有表現とエンティティタイプの抽出
text = u'My name is James from Japan. Today, talk basketball!'
doc = nlp(text)
token = doc[3] # Koya
print([d for d in doc])
print(token)

print("-------------------------------------------------------")

for d in doc:
    print((d.text, d.pos_, d.dep_))

print("-------------------------------------------------------")

print([(d.text, d.label_, spacy.explain(d.label_)) for d in doc.ents])
