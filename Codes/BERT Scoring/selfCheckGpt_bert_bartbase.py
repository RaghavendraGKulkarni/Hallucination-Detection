import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import json
import torch
import spacy
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore
import selfcheckgpt

f = open('bart_base_dataset.json')
gpt3_data = json.load(f)
f.close()

nlp = spacy.load("en_core_web_sm")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

selfcheck_bertscore = SelfCheckBERTScore(rescale_with_baseline=True)

def find_SelfCheckGPT_score(data, selfcheck_bertscore):
    results = {"mqag": [], "bertscore": [], "ngram": []}
    for i in tqdm(range(0, 100)):
        passage = data[i]['bert_text']
        sentences = [sent.text.strip() for sent in nlp(passage).sents]
        sampled_passages = data[i]['bert_text_examples']
        bert_score = selfcheck_bertscore.predict(sentences=sentences, sampled_passages=sampled_passages)
        results['bertscore'].append(bert_score.tolist())
        print(results)
    return results

# batch_size = 30 
wikibio3_scores = find_SelfCheckGPT_score(gpt3_data, selfcheck_bertscore)

with open("bart-base_bertscores.json", "w") as outfile: 
    json.dump(wikibio3_scores, outfile)