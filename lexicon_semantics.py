#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 14:08:32 2017

@author: arhumsavera
"""
from collections import defaultdict, OrderedDict
import re,math


counts=defaultdict(lambda:1) #Add 1 smoothing
polarities= defaultdict(float)
unique_words=set()
fliter_words=True
filename="tweets.txt"
prefixes = ('@', '#')
POS_WORDS=["good", "nice", "love", "excellent", "fortunate", "correct", "superior"]
NEG_WORDS=["bad", "nasty", "poor", "hate", "unfortunate", "wrong", "inferior"]
pos="POS"
neg="NEG"



def get_polarity(w):
    near_pos_key='_'.join(sorted([pos,w])) 
    near_neg_key='_'.join(sorted([neg,w]))
    n=(counts[near_pos_key]*counts[neg])
    d=(counts[pos]*counts[near_neg_key])
    return math.log(float(n)/d)
    

for line in open(filename):
    #print line
    tweet = re.sub(r"http\S+", "", line)                                    #remove urls
    splitwords=tweet.lower().split()
    filteredwords = [x for x in splitwords if not x.startswith(prefixes)]   #ignore prefix words
    poswords = [pos if x in POS_WORDS else x for x in filteredwords]        #combine positive words
    words = [neg if x in NEG_WORDS else x for x in poswords]                #combine negative words
    words=list(set(words))                                                  #remove duplicates
    #print words
    for i in range(len(words)):
        for j in range(i+1,len(words)):                                     #for each word pair
            w=[words[i],words[j]]
            k='_'.join(sorted(w))                                           #sorted tuple 'key' of word pair
            #print k
            counts[k]+=1                                                    #increase co occurence count
        counts[words[i]]+=1                                                 #increase individual occurence
        unique_words.add(words[i])                                          #add to unique list for later
            

#remove POS and NEG from word list
unique_words.remove(pos)
unique_words.remove(neg)

if fliter_words:
    for key in counts.keys():
        if counts[key]<500:
            del counts[key]


for word in unique_words:
    if word not in POS_WORDS and word not in NEG_WORDS:
        polarities[word]=get_polarity(word)

sorted_polarities=sorted(polarities.items() , key=lambda x : x[1] , reverse=True)
sorted_positive_polarities = sorted_polarities[:10]
sorted_negative_polarities= sorted_polarities[-10:]

print "---Most positive---"
for k,v in sorted_positive_polarities:
    print k,":",v
print "-------------------"
print "---Most negative---"
print "-------------------"
for k,v in sorted_negative_polarities:
    print k,":",v
