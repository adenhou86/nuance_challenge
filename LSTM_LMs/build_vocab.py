import argparse
import pickle
from tokenization import Vocab, Tokenizer

def argparser():
    p = argparse.ArgumentParser()

    # Required parameters
    p.add_argument('--corpus', default=None, type=str, required=True)
    p.add_argument('--vocab', default=None, type=str, required=True)
    
    # Other parameters
    p.add_argument('--is_tokenized', action='store_true',
                   help='Whether the corpus is already tokenized')
    p.add_argument('--tokenizer', default='spacy', type=str,
                   help='Tokenizer used for input corpus tokenization')
    p.add_argument('--unk_token', default='<unk>', type=str,
                   help='The representation for any unknown token')
    p.add_argument('--pad_token', default='<pad>', type=str,
                   help='The representation for the special token of padding token')
    p.add_argument('--bos_token', default='<bos>', type=str,
                   help='The representation for the special token of beginning-of-sequence token')
    p.add_argument('--eos_token', default='<eos>', type=str,
                   help='The representation for the special token of end-of-sequence token')
    p.add_argument('--min_freq', default=3, type=int,
                   help='The minimum frequency required for a token')
    p.add_argument('--lower', action='store_true',
                   help='Whether to convert the texts to lowercase')

    config = p.parse_args()
    return config

def main(config):
    print(config)
    
    list_of_tokens = []
    if config.is_tokenized:
        # Read tokens
        with open(config.corpus, 'r', encoding='utf-8') as reader:
            for li, line in enumerate(reader):
                list_of_tokens += line.strip().split()
    else:
        # Select tokenizer
        if config.tokenizer=='spacy':
          import spacy
          nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"]) 
          tokenizer = Tokenizer(tokenization_fn=nlp)

          # Tokenization & read tokens
          with open(config.corpus, 'r', encoding='utf-8') as reader:
            for li, line in enumerate(reader):
              if line != '\r\n':
                if config.lower:
                  list_of_tokens += tokenizer.tokenize(line.strip().lower())
                else:
                  list_of_tokens += tokenizer.tokenize(line.strip().upper())

    # Build vocabulary
    print(len(list_of_tokens), len(set(list_of_tokens)))                
    vocab = Vocab(list_of_tokens=list_of_tokens,
                  unk_token=config.unk_token,
                  pad_token=config.pad_token,
                  bos_token=config.bos_token,
                  eos_token=config.eos_token,
                  min_freq=config.min_freq,
                  lower=config.lower)
    vocab.build()
    print('Vocabulary size: ', len(vocab))

    # Save vocabulary
    with open(config.vocab, 'wb') as writer:
        pickle.dump(vocab, writer)
    print('Vocabulary saved to', config.vocab)

if __name__=='__main__':
    config = argparser()
    main(config)