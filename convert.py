import os
import argparse
from tqdm import tqdm
import pickle
import dawg
from factorizer import Factorizer
import struct

data = pickle.load(open('/nas/home/thawani/etok/factorizer/eng_256_final_vocab.pkl', 'rb'))
vocab = {}
# create dawg file based on factorizer.py
for (r,b,g),(w,p) in tqdm(data.items()):
    # if bos in beginning of word, replace it
    if len(w) > 0:
        if w[0] == ' ':
            w = '܂׀ԅ' + w[1:] #'⸥' + w[1:]
        if w[-1] == ' ':
            w = w[:-1] + '܂׀Ԅ'#'⸤'
    #if w in vocab:
    #    print(f"Duplicate word: {w=} {p=} {r=} {g=} {b=} {vocab[w]=}")
    if w in vocab and vocab[w][0] < p: # zi = argmaxz p(wi|z)
        continue
    assert type(p) == float and type(r) == int and type(g) == int and type(b) == int
    vocab[w] = (p, r, g, b)
    #vocab[w] = (p,r.to_bytes(1, 'big'), g.to_bytes(1, 'big'), b.to_bytes(1, 'big'))
    # convert to unsigned char
    
#words = [(word, struct.pack("fBBB", p, r, g, b)) for word, (p, r, g, b) in vocab.items()]
#words = [(word, (p, r.to_bytes(1, 'big'), g.to_bytes(1, 'big'), b.to_bytes(1, 'big'))) for word, (p, r, g, b) in vocab.items()]
#words = [(word, (p, int.from_bytes(r, byteorder='big'), int.from_bytes(g, byteorder='big'), int.from_bytes(b, byteorder='big'))) for word, (p, r, g, b) in vocab.items()]
d = dawg.RecordDAWG("fBBB", vocab.items(),payload_separator=b'\xff')
d.save('sym_en_256_final_vocab.dawg')
print("Saved dawg")

# self.id_to_subword = np.zeros((256, 256, 256), dtype=object)
# for subword, (_, index_1, index_2, index_3) in self.trie.iteritems():
#    self.id_to_subword[index_1, index_2, index_3] = bytes_to_subword(subword)

tokenizer = Factorizer("sym_en_256_final_vocab.dawg")
sentence = "The echo of a distant time comes willowing across the sand, and everything is green and submarine."

encoding = tokenizer(sentence)

print(f"INPUT:    {sentence}")
print(f"SUBWORDS: {' '.join(encoding.tokens)}")
print(f"INDICES:  {' '.join(str(index) for index in encoding.ids)}")
print(f"DECODED:  {tokenizer.decode(encoding.ids)}")