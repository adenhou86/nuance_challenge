#!/usr/bin/env python3 
# to launch it: cat <text file> | tok.py 

import sys 
import spacy


nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"]) 

cleaned_txt = []

for line in sys.stdin:
    if line != '\r\n':
        words = [tok.text for tok in nlp(line.strip().upper())]
        tokenized_sentence = words
        cleaned_sentence = " ".join(words)
        #print(f"line : {line}")
        #print(f"tokenized_sentence : {tokenized_sentence}")
        #print(f"cleaned_sentence : {cleaned_sentence}") 
        
        cleaned_txt.append(cleaned_sentence)
        
#print(f"cleaned_txt : {cleaned_txt}")
print("\n".join(cleaned_txt))