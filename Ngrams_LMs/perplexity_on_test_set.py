#!/usr/bin/env python3

import kenlm 
import argparse 
import sys 
import math

modelName = sys.argv[1]
model = kenlm.Model(modelName)

list_ppx = []

for i,line in enumerate(sys.stdin,1):
    if i > 10:
        break
    else:
        sentence = line.strip()
        sum_inv_logs = sum(prob for prob, _, _ in model.full_scores(sentence))
        n = len(sentence.split())
        print(f"sentence : {sentence} --> Proba {sum_inv_logs}")
        list_ppx.append(sum_inv_logs)
        """
        words = len(sentence.split()) + 1# For </s>
        ppx = model.perplexity(sentence)
        print(f"sentence : {sentence} --> PPL {ppx}")
        list_ppx.append(ppx)
        """
#print(sum(list_ppx), sum(list_ppx)/(len(list_ppx)+1), list_ppx)        
print(math.pow(10.0, sum(list_ppx)))