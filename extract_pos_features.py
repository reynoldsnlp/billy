"""POS-tag and extract features from corpus of 18th cent. female poets."""

from glob import glob
import re
import sys

import nltk


try:
    corp_glob = sys.argv[1]
except IndexError:
    corp_glob = 'WomenPoetsAll/corpus/*.txt'
output_name = '_'.join(corp_glob.split('/')[:-1]) + '.csv'

coords = set('for and nor but or yet so'.split())
yearRE = re.compile(r'_([0-9]{4})_')

tags = '''CC
CD
DT
EX
FW
IN
JJ
JJR
JJS
LS
MD
NN
NNS
NNP
NNPS
PDT
POS
PRP
PRP$
RB
RBR
RBS
RP
SYM
TO
UH
VB
VBD
VBG
VBN
VBP
VBZ
WDT
WP
WP$
WRB'''.split()

with open(output_name, 'w') as output_file:
    print(*'author title words/sent'.split() + list(sorted(tags)) + ['year'],
          sep='\t', file=output_file)
    for f in glob(corp_glob):
        print('processing {} ...'.format(f))
        with open(f) as input_file:
            author, year, title = f.split('_', maxsplit=2)
            author = author.split('/')[-1]
            title = title.rstrip('.txt')
            orig_txt = input_file.read()
            tokens = nltk.word_tokenize(orig_txt)
            tokens = [w.lower() for w in tokens]
            N = len(tokens)  # total number of words
            sents = len(nltk.sent_tokenize(orig_txt))  # total number of sentences
            tagged = nltk.pos_tag(tokens)
            # list comprehensions to filter the tagged text
            pos_dict = {}
            for t in tags:
                pos_dict[t] = len([1 for tok, tag in tagged if tag == t]) / N
            num_coords = len([1 for tok, tag in tagged if tok in coords])
        print(author, title, N / sents, *[f for tag, f in sorted(pos_dict.items())],
              year, sep='\t', file=output_file)
