import os
import argparse
from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import Dataset
from model import Model

torch.backends.cuda.matmul.allow_tf32 = True
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def run(model, dataset, word: str, bow=True, eow=True, prob_keep=1.0):
    input = torch.tensor(dataset.numericalize(dataset.encode(word)[0], bow=bow, eow=eow, prob_keep=prob_keep)).unsqueeze(0).cuda()
    prediction, indices, perplexity = model.inference(input)

    gold = dataset.ids_to_word(input[0, 1:-1].tolist())
    prediction = dataset.ids_to_word(prediction[0, :].tolist())

    print(f"{gold}\t-> {indices[0][0].item()}, {indices[1][0].item()}, {indices[2][0].item()} -> {prediction}, {(-perplexity).exp().item()*100.0:.2f}%")


#bin_file = '/nas/ckgfs/users/thawani/etok/factorizer_models/eng_regularized_256_42_smooth-breeze-41.bin' # 'num_256_42_sparkling-snowball-62.bin' # eng_regularized_256_42_smooth-breeze-41.bin
bin_file = r'/home/harsh/AI-Projects/factorizer/avi_files/backup_wt0.1_bs1e13_lr1e-4.bin'
train_file = r'/home/harsh/AI-Projects/factorizer/output.tsv' # '/nas/ckgfs/users/thawani/etok/count_1w_train.txt' # '/nas/ckgfs/users/thawani/etok/num_factorizer_data/numtrain.txt' #
out_file = 'ar_256_final_vocab.pkl' # 'eng_256_final_vocab.pkl' # 'num_256_final_vocab.pkl' #

bin = torch.load(bin_file)
train_dataset = Dataset(bin['args'], path=train_file, vocab=None, random=True, split=True) # 300k

model = Model(bin['args'], train_dataset).to('cuda:0')
model.load_state_dict(bin['model'])
model.eval()
run(model, train_dataset, "123", bow=True, eow=True, prob_keep=1.0)

def decode(code):
    ids = model.decode(torch.tensor([code]).to('cuda:0'))[0]
    bseq = [bin['vocabulary'][i] for i in ids[0]]
    # trim BOW and EOW if present
    decoded = bytes([b for b in bseq if b not in ['[BOW]','[EOW]']]).decode('utf-8')
    if '[EOW]' in bseq:
        decoded+= ' '
    if '[BOW]' in bseq:
        decoded = ' ' + decoded

def batch_decode(codes):
    breakpoint()
    ids, ppls = model.decode(torch.tensor(codes)[:10].to('cuda:0'))
    bseq = [[bin['vocabulary'][i] for i in id] for id in ids]
    decoded = []
    for b in bseq:
        # trim BOW and EOW if present
        try:
            decoded.append(bytes([b for b in b if b not in ['[BOW]','[EOW]']]).decode('utf-8'))
        except:
            decoded.append('')
        if '[EOW]' in b:
            decoded[-1] +=  ' '
        if '[BOW]' in b:
            decoded[-1] = ' ' + decoded[-1]
    return decoded, ppls

final_vocab = {}
for r in tqdm(range(0,256)):
    #for g in range(0,256):
        #for b in range(0,256):
        #    code = [r,g,b]
        #   final_vocab[tuple(code)] = decode(code)
        # codes = [[r,g,b] for b in range(0,256)]
        # for b, d in zip(range(0,256), decoded):
        #final_vocab[(r,g,b)] = d
    
    codes = [[r,g,b] for g in range(0,256) for b in range(0,256)]
    decoded, ppls = batch_decode(codes)
    for g in range(0,256):
        for b, d, p in zip(range(0,256), decoded[g*256:(g+1)*256], ppls[g*256:(g+1)*256]):
            final_vocab[(r,g,b)] = (d,p)
            
with open(out_file, 'wb') as f:
    pickle.dump(final_vocab, f)
print("Done")

# while True:
#    word = input("Enter word: ")
#    run(model, train_dataset, word, bow=True, eow=True, prob_keep=1.0)