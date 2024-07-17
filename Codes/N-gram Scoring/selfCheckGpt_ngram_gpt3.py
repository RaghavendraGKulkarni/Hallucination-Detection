import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import json
import torch
import spacy
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from selfcheckgpt.modeling_selfcheck import SelfCheckNgram
import selfcheckgpt

f = open('gpt3_wikibio.json')
gpt3_data = json.load(f)
f.close()

nlp = spacy.load("en_core_web_sm")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

selfcheck_ngram = SelfCheckNgram(n=1)

def find_SelfCheckGPT_score(data, selfcheck_bertscore):
    results = {"mqag": [], "bertscore": [], "ngram": []}
    for i in tqdm(range(0, len(data))):
        passage = data[i]['gpt3_text']
        sentences = data[i]['gpt3_sentences']
        sampled_passages = data[i]['gpt3_text_samples']
        ngram_score = selfcheck_ngram.predict(sentences = sentences, passage = passage, sampled_passages = sampled_passages)
        results['ngram'].append(ngram_score)
    return results

# batch_size = 30 
wikibio3_scores = find_SelfCheckGPT_score(gpt3_data, selfcheck_ngram)

with open("gpt3_ngram_scores.json", "w") as outfile: 
    json.dump(wikibio3_scores, outfile)