import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tokenization import Tokenizer, Vocab
from dataset_utils import Corpus
from models import LSTMLM, BiLSTMLM

def argparser():
    p = argparse.ArgumentParser()

    # Required parameters
    p.add_argument('--corpus', default=None, type=str, required=True)
    p.add_argument('--vocab', default=None, type=str, required=True)
    p.add_argument('--model', default=None, type=str, required=True)
    p.add_argument('--model_type', default=None, type=str, required=True,
                   help='Model type selected in the list: LSTM')
    
    # Input parameters
    p.add_argument('--is_tokenized', action='store_true',
                   help='Whether the corpus is already tokenized')
    p.add_argument('--tokenizer', default='spacy', type=str,
                    help='Tokenizer used for input corpus tokenization')
    p.add_argument('--max_seq_len', default=32, type=int,
                   help='The maximum total input sequence length after tokenization')
    p.add_argument('--generate_sentences', action='store_true',
                   help='Show the genereate sentences for test set instead of showing the perplexity')

    # Inference parameters
    p.add_argument('--multi_gpu', action='store_true',
                   help='Whether to inference with multiple GPU')
    p.add_argument('--cuda', default= True, type=bool,
                   help='Whether CUDA is cureently available')
    p.add_argument('--batch_size', default=16, type=int,
                   help='Batch size for inference')

    # Model parameters
    p.add_argument('--embedding_size', default=256, type=int,
                   help='Word embedding vector dimension')
    p.add_argument('--hidden_size', default=1024, type=int,
                   help='Hidden size of LSTM')
    p.add_argument('--n_layers', default=3, type=int,
                   help='Number of layers in LSTM')
    p.add_argument('--dropout_p', default=.2, type=float,
                   help='Dropout rate used for dropout layer in LSTM')

    config = p.parse_args()
    return config

def sentence_from_indexes(indexes):
    # |indexes| = (max_seq_len-1)
    
    # Convert indexes to tokens
    tokens = tokenizer.inverse_transform(indexes.tolist())
    # |tokens| = (max_seq_len-1)
    
    try:
        # Return tokens up to eos_token
        first_eos_token_index = tokens.index(vocab.eos_token)
        return ' '.join(tokens[:first_eos_token_index])
    except ValueError:
        # Only if eos_token is not in the token list
        return ' '.join(tokens)

def inference():
    n_batches, n_samples = len(loader), len(loader.dataset)
    
    model.eval()
    total_loss, total_ppl = 0, 0

    with torch.no_grad():
        for iter_, batch in enumerate(loader):
            inputs, targets = batch
            # |inputs|, |targets| = (batch_size, seq_len), (batch_size, seq_len)

            preds = model(inputs)
            # |preds| = (batch_size, max_seq_len-1, len(vocab))

            if config.multi_gpu:
                # If the model run parallelly using DataParallelModel,the output tensor size is as follows.
                # |preds| = [(batch_size/n_gpus, max_seq_len-1, len(vocab))] * n_gpus                

                # So, concatenate tensors split by multi-gpu usage
                preds = torch.cat([pred for pred in preds], dim=0)
                # |preds| = (batch_size, max_seq_len-1, len(vocab))

            if config.generate_sentences:
              # Returns the largest element of the predictions
              topv, topi = torch.topk(preds, 1)
              # |topv|, |topi| = (batch_size, max_seq_len-1, 1)
              
              # Convert indexes to sentences
              for i, (each_topi, target) in enumerate(zip(topi, targets)):
                  target_sentences = sentence_from_indexes(target)
                  pred_sentences = sentence_from_indexes(each_topi.squeeze(-1))
                  
                  print('#{} =============='.format(iter_*config.batch_size + i))
                  print('Actu:\t{}\nPred:\t{}\n'.format(target_sentences, pred_sentences))
            else:
              preds = preds.view(-1, len(vocab)).contiguous()    
              # |preds| = (batch_size*seq_len, len(vocab))
              
              targets = targets.view(-1).contiguous()
              # |targets| = (batch_size*seq_len)
              
              loss = loss_fn(preds, targets)
              total_loss += loss.item()
              total_ppl += np.exp(loss.item())
        if !config.generate_sentences:
            print('====> Test : Average loss: {:.4f} \tPerplexity: {:.5f}'.format(
            total_loss/n_batches, total_ppl/n_batches))
            print(f"n_batches : {n_batches}, n_samples : {n_samples}")

if __name__=='__main__':
    config = argparser()
    print(config)

    # Load vocabulary
    import pickle
    with open(config.vocab, 'rb') as reader:
        vocab = pickle.load(reader)
    
    # Select tokenizer
    if config.tokenizer=='spacy':
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"]) 
        tokenizer = Tokenizer(tokenization_fn=nlp,
                              vocab=vocab, max_seq_length=config.max_seq_len)

    # Build dataloader
    corpus = Corpus(corpus_path=config.corpus, tokenizer=tokenizer, model_type=config.model_type, cuda=config.cuda)
    loader = DataLoader(dataset=corpus, batch_size=config.batch_size)

    # Load model with trained parameters
    if config.model_type=='LSTM':
        model = LSTMLM(input_size=len(vocab),
                       embedding_size=config.embedding_size,
                       hidden_size=config.hidden_size,
                       output_size=len(vocab),
                       n_layers=config.n_layers,
                       dropout_p=config.dropout_p)
    elif config.model_type=='BiLSTM':
        model = BiLSTMLM(input_size=len(vocab),
                         embedding_size=config.embedding_size,
                         hidden_size=config.hidden_size,
                         output_size=len(vocab),
                         n_layers=config.n_layers,
                         dropout_p=config.dropout_p)

    loss_fn = nn.NLLLoss(ignore_index=vocab.stoi[vocab.pad_token])
    
    if config.cuda:
        if config.multi_gpu:
            from parallel import DataParallelModel, DataParallelCriterion
            model = DataParallelModel(model).cuda()
            loss_fn = DataParallelCriterion(loss_fn).cuda()
        else:
            model = model.cuda()
            loss_fn = loss_fn.cuda()

    if config.cuda:
        if config.multi_gpu:
            from parallel import DataParallelModel, DataParallelCriterion
            model = DataParallelModel(model).cuda()
            loss_fn = DataParallelCriterion(loss_fn).cuda()
        else:
            model = model.cuda()
            loss_fn = loss_fn.cuda()
    model.load_state_dict(torch.load(f"./models_logs/{config.model}"))
    print('=========MODEL=========\n',model)

    # Inference
    inference()