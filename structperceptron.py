from __future__ import division
import sys,re,random
from collections import defaultdict
from pprint import pprint


import vit_starter 
OUTPUT_VOCAB = set(""" ! # $ & , @ A D E G L M N O P R S T U V X Y Z ^ """.split())
#OUTPUT_VOCAB = set(""" ! # $ """.split())


# Utilities

def dict_subtract(vec1, vec2):
    """ vec1 and vec2 as dict representations of sparse vectors"""
    out = defaultdict(float)
    out.update(vec1)
    for k in vec2: out[k] -= vec2[k]
    return dict(out)

def dict_argmax(dct):
    """Return the key whose value is largest. In other words: argmax_k dct[k]"""
    return max(dct.iterkeys(), key=lambda k: dct[k])

def dict_dotprod(d1, d2):
    """Return the dot product (aka inner product) of two vectors"""
    smaller = d1 if len(d1)<len(d2) else d2  
    total = 0
    for key in smaller.iterkeys():
        total += d1.get(key,0) * d2.get(key,0)
    return total

def read_tagging_file(filename):
    """Returns list of sentences from a two-column formatted file.
    Each returned sentence is the pair (tokens, tags) where each of those is a
    list of strings.
    """
    sentences = open(filename).read().strip().split("\n\n")
    ret = []
    for sent in sentences:
        lines = sent.split("\n")
        pairs = [L.split("\t") for L in lines]
        tokens = [tok for tok,tag in pairs]
        tags = [tag for tok,tag in pairs]
        ret.append( (tokens,tags) )
    return ret
###############################


def do_evaluation(examples, weights):
    num_correct,num_total=0,0
    for tokens,goldlabels in examples:
        N = len(tokens); assert N==len(goldlabels)
        predlabels = predict_seq(tokens, weights)
        num_correct += sum(predlabels[t]==goldlabels[t] for t in range(N))
        num_total += N
    print "%d/%d = %.4f accuracy" % (num_correct, num_total, num_correct/num_total)
    return num_correct/num_total

def fancy_eval(examples, weights):
    confusion = defaultdict(float)
    bygold = defaultdict(lambda:{'total':0,'correct':0})
    for tokens,goldlabels in examples:
        predlabels = predict_seq(tokens, weights)
        for pred,gold in zip(predlabels, goldlabels):
            confusion[gold,pred] += 1
            bygold[gold]['correct'] += int(pred==gold)
            bygold[gold]['total'] += 1
    goldaccs = {g: bygold[g]['correct']/bygold[g]['total'] for g in bygold}
    for gold in sorted(goldaccs, key=lambda g: -goldaccs[g]):
        print "gold %s acc %.4f (%d/%d)" % (gold,
                goldaccs[gold],
                bygold[gold]['correct'],bygold[gold]['total'],)

def show_predictions(tokens, goldlabels, predlabels):
    print "%-20s %-4s %-4s" % ("word", "gold", "pred")
    print "%-20s %-4s %-4s" % ("----", "----", "----")
    for w, goldy, predy in zip(tokens, goldlabels, predlabels):
        out = "%-20s %-4s %-4s" % (w,goldy,predy)
        if goldy!=predy:
            out += "  *** Error"
        print out



def train(examples, stepsize=1, numpasses=10, do_averaging=False, devdata=None):
    """
    Train a structured perceptron
    """

    weights = defaultdict(float)
    S = defaultdict(float)
    t = 0

    def get_averaged_weights():
        # IMPLEMENT ME!
        avg_weights=defaultdict(float)
        for feature,val in S.iteritems():
            avg_weights[feature]=weights[feature]-(val/t)
        return avg_weights

    for pass_iteration in range(numpasses):
        print "Training iteration %d" % pass_iteration
        for tokens,goldlabels in examples:
            predlabels=predict_seq(tokens, weights)
            if predlabels != goldlabels:
                
                fvec = dict_subtract(features_for_seq(tokens, goldlabels), features_for_seq(tokens, predlabels))
                for feature,value in fvec.iteritems():
                    weights[feature] = weights[feature] + (stepsize*value)
                    S[feature] = S[feature] + ((t-1)*stepsize *value)
                t+=1
            
        


        # Evaluation at the end of a training iter
        print "TR  RAW EVAL:",
        #do_evaluation(examples, weights)
        if devdata:
            print "DEV RAW EVAL:",
         #   do_evaluation(devdata, weights)
        if devdata and do_averaging:
            print "DEV AVG EVAL:",
            do_evaluation(devdata, get_averaged_weights())

    print "Learned weights for %d features from %d examples" % (len(weights), len(examples))

    return weights if not do_averaging else get_averaged_weights()

def predict_seq(tokens, weights):
    """
    takes tokens and weights, calls viterbi and returns the most likely
    sequence of tags
    """
    # once you have Ascores and Bscores, could decode with
    # predlabels = greedy_decode(Ascores, Bscores, OUTPUT_VOCAB)
    Ascores, Bscores=calc_factor_scores(tokens, weights)
    pred_seq=vit_starter.viterbi(Ascores, Bscores, OUTPUT_VOCAB)
    #print pred_seq
    return pred_seq




def local_emission_features(t, tag, tokens):
    """
    Feature vector for the B_t(y) function
    t: an integer, index for a particular position
    tag: a hypothesized tag to go at this position
    tokens: the list of strings of all the word tokens in the sentence.
    """
    curword = tokens[t]
    feats = {}
    feats["tag=%s_biasterm" % tag] = 1
    feats["tag=%s_curword=%s" % (tag, curword)] = 1
    
    
    import re
    if curword.startswith("@"):
        feats["tag=%s_@=%s" % (tag, curword)]= 1 
    if curword.startswith("#"):
        feats["tag=%s_#=%s" % (tag, curword)]= 1 
    if "http" in curword:
        feats["tag=%s_url=%s" % (tag, curword)]= 1 #else 0
        
    feats["tag=%s_firstcapital"%tag] = 1 if(re.findall('^[A-Z]', curword)) else 0
    
    

    return feats

def features_for_seq(tokens, labelseq):
    """
    tokens: a list of tokens
    labelseq: a list of output labels
    The full f(x,y) function. Returns one big feature vector.

    This returns a feature vector represented as a dictionary.
    """
    fvec=defaultdict(float)
    for t in range(len(tokens)):
        if t!=0:
            fvec["trans_%s_%s"% (labelseq[t - 1], labelseq[t])] += 1
        local_features=local_emission_features(t,labelseq[t],tokens)
        for k,v in local_features.iteritems():
            fvec[k]+=v
            
    
    return fvec

def calc_factor_scores(tokens, weights):
    """
    tokens: a list of tokens
    weights: perceptron weights (dict)

    returns a pair of two things:
    Ascores which is a dictionary that maps tag pairs to weights
    Bscores which is a list of dictionaries of tagscores per token
    """
    N = len(tokens)
    # MODIFY THE FOLLOWING LINE
    Ascores = { (tag1,tag2): weights["trans_%s_%s"%(tag1,tag2)] for tag1 in OUTPUT_VOCAB for tag2 in OUTPUT_VOCAB }
    Bscores = [defaultdict(float) for x in range(N)]
    for t in range(N):
        for tag in OUTPUT_VOCAB:
            #print tag
            #print local_emission_features(t, tag, tokens)
            Bscores[t][tag] += dict_dotprod(weights, local_emission_features(t, tag, tokens))
            #print Bscores[t][tag]
            #print "---"
    assert len(Bscores) == N
    return Ascores, Bscores


