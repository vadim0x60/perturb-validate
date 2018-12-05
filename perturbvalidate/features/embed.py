import nltk
from nltk.tokenize import TweetTokenizer
import sentencepiece as spm
import numpy as np
from gensim.models import KeyedVectors
import os
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
bpe_model_location = os.path.join(project_dir, 'data', 'external', 'ru.wiki.bpe.vs10000.model')
bpe_vec_location = os.path.join(project_dir, 'data', 'external', 'ru.wiki.bpe.vs10000.d100.w2v.bin')
alg_location = os.path.join(project_dir, 'data', 'raw', 'alg.txt')
testament_location = os.path.join(project_dir, 'data', 'raw', 'testament.txt')

tokenizer = TweetTokenizer()
tagger = nltk.tag._get_tagger('rus')
pos_tags = list(tagger.classes)

sp = spm.SentencePieceProcessor()
sp.Load(bpe_model_location)
bpe_model = KeyedVectors.load_word2vec_format(bpe_vec_location, binary=True)

def tokenize(text):
    sentences = []

    for sent in nltk.sent_tokenize(text, language='russian'):
        tokens = tokenizer.tokenize(sent)
        if tokens[0].isdigit():
            tokens = tokens[1:]
        sentences.append(tokens)

    return sentences

def bpe_embed(text):
    pieces = sp.encode_as_pieces(text)
    embedding = np.zeros(bpe_model.vector_size)
    piece_count = 0

    for piece in pieces:
        try:
            embedding += bpe_model[piece]
            piece_count += 1
        except KeyError:
            pass

    if piece_count:
        embedding /= piece_count
            
    return embedding

def embed_sentences(sentences):
    for sentence in tagger.tag_sents(sentences):
        sent_embedding = []
    
        for token, pos in sentence:
            token_embedding = bpe_embed(token)
            pos_embedding = np.zeros(len(pos_tags))
            pos_embedding[pos_tags.index(pos)] = 1
            sent_embedding.append(np.concatenate([token_embedding, pos_embedding]))
        
        yield sent_embedding