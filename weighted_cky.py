from pprint import pprint
from collections import defaultdict

grammar_rules = []
lexicon = {}
probabilities = defaultdict(float)
possible_parents_for_children = {}
symbol_list=set()
backpointers={}


def populate_grammar_rules():
    global grammar_rules, lexicon, probabilities, possible_parents_for_children,symbol_list
    filename='pcfg_grammar_modified'
    for line in tuple(open(filename, 'r')):
        #print "got ", line
        if not line.strip(): continue
        left, right = line.split("->")
        left = left.strip()
        children = right.split()
        prob=children[-1]
        children=children[:-1]
        probabilities[tuple([left]+children)]=float(prob)
        rule = (left, tuple(children))
        if len(children)>1:
            grammar_rules.append(rule)
        else:
            if left not in lexicon:
                lexicon[left]=set()
            lexicon[left].add(children[0])
    possible_parents_for_children = {}
    for parent, production in grammar_rules:
        leftchild, rightchild=production
        symbol_list.add(leftchild)
        symbol_list.add(rightchild)
        if (leftchild, rightchild) not in possible_parents_for_children:
            possible_parents_for_children[leftchild, rightchild] = []
        possible_parents_for_children[leftchild, rightchild].append(parent)
        
    # Error checking   
    all_parents = set(x[0] for x in grammar_rules) | set(lexicon.keys())
    for par, (leftchild, rightchild) in grammar_rules:
        if leftchild not in all_parents:
            assert False, "Nonterminal %s does not appear as parent of prod rule, nor in lexicon." % leftchild
        if rightchild not in all_parents:
            assert False, "Nonterminal %s does not appear as parent of prod rule, nor in lexicon." % rightchild
            
    print "Grammar rules in tuple form:"
    pprint(grammar_rules)
    print "Rule parents indexed by children:"
    pprint(possible_parents_for_children)
    print "probabilities"
    pprint(probabilities)
    print "Lexicon"
    pprint(lexicon)



def get_symbol_from_det(word):
    global lexicon, probabilities
    #print "get symbol from det got", word
    symbols=[]
    for symbol,dets in lexicon.iteritems():
        #print "looking at symbol", symbol
        if word in dets:
            symbols.append(symbol)
    return symbols


def get_symbol_from_pairs(s1,s2):
    global possible_parents_for_children
    if (s1,s2) in possible_parents_for_children:
        return possible_parents_for_children[s1,s2]
    return [] #No possible parents


def get_tree_probability(i,j,production,symbol):
    if j-i==1:
        #print "base:" ,production
        #print "returning ",probabilities[symbol,production]
        return probabilities[symbol,production]
    else:
        #print "PRINTING ",production
        k=production[0]
        left=production[1]
        right=production[2]
        
        
        if k-i==1:
            leftprob= get_tree_probability(i,k,backpointers[i,k][left],left)
            #print "base left got ", leftprob
        else:
            leftprob=get_tree_probability(i,k,backpointers[i,k][left][0],left)
            #print "regular left got ", leftprob

        if j-k==1:
             rightprob= get_tree_probability(k,j,backpointers[k,j][right],right)
             #print "base right got ", rightprob

        else:
            rightprob=get_tree_probability(k,j,backpointers[k,j][right][0],right)
            #print "regular right got ", rightprob

        
        ruleprob=probabilities[symbol,left,right]
        #print "rule prob for ",symbol," to ",left," ",right," is ", ruleprob
        #print "left sub prob and right sub prob are: ",leftprob,rightprob
        #print "returning :",ruleprob*leftprob*rightprob
        return ruleprob*leftprob*rightprob




def traverse(i,j,production,symbol):
    if j-i==1:
        #print "base:" ,production
        return [symbol,production]
    else:
        #print "PRINTING ",production
        k=production[0]
        left=production[1]
        right=production[2]
        
        if k-i==1:
            leftlist=traverse(i,k,backpointers[i,k][left],left)
        else:
            leftlist=traverse(i,k,backpointers[i,k][left][0],left)

        if j-k==1:
            rightlist=traverse(k,j,backpointers[k,j][right],right)
        else:
            rightlist=traverse(k,j,backpointers[k,j][right][0],right)

        return[symbol, [leftlist,rightlist]]
    
def pcky_parse(sentence):
    global grammar_rules, lexicon, probabilities, possible_parents_for_children,symbol_list,backpointers
    backpointers={}    
    N = len(sentence)
    cells = {}
    
    for i in range(N):
        for j in range(i + 1, N + 1):
            cells[(i, j)] = []
            backpointers[i,j]={}
            
    for spanLen in range(1,N+1):
        for i in range(0,N-spanLen+1):
            j=i+spanLen
            if spanLen==1:
                #print ("in spanLen 1: ",i,j)
                word=sentence[i:j][0]
                symbols=get_symbol_from_det(word)
                #print "got ", symbols," for ",word
                for symbol in symbols:
                    cells[i,j]+=[symbol]
                    if symbol not in backpointers[i,j]:
                        backpointers[i,j][symbol]=[]
                    backpointers[i,j][symbol]=word
                #print"cell",i,j,"is now ",cells[i,j]
            else:
                for k in range(i+1,j):
                    #print("spanLen", spanLen,"values ",i,j," k:",k)
                    #print "need cells: ",cells[i,k], " and ",cells[k,j]
                    if cells[i,k] and cells[k,j]:
                        s1set=cells[i,k] #get the whole list of symbols that matched up
                        s2set=cells[k,j]
                        if s1set and s2set:
                            #print("retrieved symbols:",s1set,s2set)
                            if not isinstance( s1set,list):
                                s1set=[s1set]
                            if not isinstance( s2set,list):
                                #print "converting ", s2set
                                s2set=[s2set] 
                            #print("calling for ",s1set[0],s2set[0])
                            for s1 in s1set:
                                for s2 in s2set:
                                    #print("calling for ",s1,s2)
                                    symbols=get_symbol_from_pairs(s1,s2)
                                    #print"got ",symbols
                                    if symbols:
                                        for symbol in symbols:
                                            cells[i,j]+=[symbol]
                                            if symbol not in backpointers[i,j]:
                                                backpointers[i,j][symbol]=[]
                                            backpointers[i,j][symbol].append((k,s1,s2))
                                            #print"cell",i,j,"is now ",cells[i,j]
    
    #pprint(cells)
    #print "backpointers:"
    #pprint(backpointers)
    
    if 'S' in cells[0,N]:
        maxprob=0
        for production in backpointers[0,N]['S']:
            split=production[0]
            treeprob=get_tree_probability(0,N,production,'S')
            if treeprob>maxprob:
                maxprob=treeprob
                bestsplit=split
            #print "--------done traversal got", treeprob
        #print "best split at ", bestsplit
        #print "printing tree"
        for production in backpointers[0,N]['S']:
            if production[0]==bestsplit:
                print "Probability:",maxprob
                return traverse(0,N,production,'S')
    return None

