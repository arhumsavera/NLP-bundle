from pprint import pprint

# The productions rules have to be binarized.
#'''
grammar_text = """
S -> NP_S VP_S
S -> NP_P VP_P
NP_S -> Det Noun_S
NP_P -> Det Noun_P
VP_S -> Verb_S NP_S
VP_S -> Verb_S NP_P
VP_P -> Verb_P NP_S
VP_P -> Verb_P NP_P
PP -> Prep NP_S
PP -> Prep NP_P
NP_S -> NP_S PP
NP_P -> NP_P PP
VP_S -> VP_S PP
VP_P -> VP_P PP
"""

lexicon = {
    'Noun_S': set(['cat', 'dog', 'table', 'food']),
    'Noun_P':set(['cats','dogs']),
    'Verb_S': set(['attacked', 'saw', 'loved', 'hated','attacks']),
    'Verb_P':set(['attack']),
    'Prep': set(['in', 'of', 'on', 'with']),
    'Det': set(['the', 'a']),
}
'''
grammar_text = """
S -> NP VP
NP -> Det Noun
VP -> Verb NP
PP -> Prep NP
NP -> NP PP
VP -> VP PP
"""

lexicon = {
    'Noun': set(['cat', 'dog', 'table', 'food']),
    'Verb': set(['attacked', 'saw', 'loved', 'hated']),
    'Prep': set(['in', 'of', 'on', 'with']),
    'Det': set(['the', 'a']),
}
#'''
# Process the grammar rules.  You should not have to change this.
grammar_rules = []
for line in grammar_text.strip().split("\n"):
    if not line.strip(): continue
    left, right = line.split("->")
    left = left.strip()
    children = right.split()
    rule = (left, tuple(children))
    grammar_rules.append(rule)
possible_parents_for_children = {}
for parent, (leftchild, rightchild) in grammar_rules:
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


backpointers={}
# print "Grammar rules in tuple form:"
# pprint(grammar_rules)
# print "Rule parents indexed by children:"
# pprint(possible_parents_for_children)


def get_symbol_from_det(word):
    global lexicon
    symbols=[]
    for symbol,dets in lexicon.iteritems():
        if word in dets:
            symbols.append(symbol)
    return symbols

def get_symbol_from_pairs(s1,s2):
    for children,parent in possible_parents_for_children.iteritems():
        if (s1,s2)==children:
            return parent
    return [] #No possible parents




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



def cky_acceptance(sentence):
    # return True or False depending whether the sentence is parseable by the grammar.
    global grammar_rules, lexicon

    # Set up the cells data structure.
    # It is intended that the cell indexed by (i,j)
    # refers to the span, in python notation, sentence[i:j],
    # which is start-inclusive, end-exclusive, which means it includes tokens
    # at indexes i, i+1, ... j-1.
    # So sentence[3:4] is the 3rd word, and sentence[3:6] is a 3-length phrase,
    # at indexes 3, 4, and 5.
    # Each cell would then contain a list of possible nonterminal symbols for that span.
    N = len(sentence)
    cells = {}
    for i in range(N):
        for j in range(i + 1, N + 1):
            cells[(i, j)] = []
    
    for spanLen in range(1,N+1):
        for i in range(0,N-spanLen+1):
            j=i+spanLen
            if spanLen==1:
                #print ("in spanLen 1: ",i,j)
                word=sentence[i:j][0] #[0] because the slice returns it as an array
                #print "word is ",word
                cells[i,j]=get_symbol_from_det(word)
            else:
                for k in range(i+1,j):
                    #print("spanLen", spanLen,"values ",i,j," k:",k)
                    s1set=cells[i,k] #get the whole list of symbols that matched up
                    s2set=cells[k,j]
                    #print("symbols:",s1set,s2set)
                    for s1 in s1set:
                        for s2 in s2set: #loops wont run if either of them are empty aka no symbol rule(None)
                            #print("calling for ",s1,s2)
                            symbol=get_symbol_from_pairs(s1,s2)
                            #print"got ",symbol
                            cells[i,j]+=symbol
                    #print"cell",i,j,"is now ",cells[i,j]
    pprint(cells)
    return 'S' in cells[0,N]



def cky_parse(sentence):
    # Return one of the legal parses for the sentence.
    # If nothing is legal, return None.
    # This will be similar to cky_acceptance(), except with backpointers.
    global grammar_rules, lexicon,backpointers
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

    # TODO replace the below with an implementation
    if 'S' in cells[0,N]:
        return traverse(0,N,backpointers[0,N]['S'][0],'S')
        #return cells[0,N]
    return None

# print cky_acceptance(['the','cat','attacked','the','food'])
# pprint( cky_parse(['the','cat','attacked','the','food']))
# pprint( cky_acceptance(['the','the']))
# pprint( cky_parse(['the','the']))
# print cky_acceptance(['the','cat','attacked','the','food','with','a','dog'])
# pprint( cky_parse(['the','cat','attacked','the','food','with','a','dog']) )
# pprint( cky_parse(['the','cat','with','a','table','attacked','the','food']) )
#
