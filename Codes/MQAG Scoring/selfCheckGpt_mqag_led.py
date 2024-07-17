import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import json
import torch
import spacy
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from selfcheckgpt.modeling_selfcheck import SelfCheckMQAG
import selfcheckgpt


f = open('led_dataset.json')
gpt3_data = json.load(f)
f.close()

nlp = spacy.load("en_core_web_sm")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

selfcheck_mqag = SelfCheckMQAG(device=device)

def find_SelfCheckGPT_score(data, selfcheck_bertscore):
    results = {"mqag": [], "bertscore": [], "ngram": []}
    for i in tqdm(range(20, 30)):
        passage = data[i]['t5_text']
        sentences = [sent.text.strip() for sent in nlp(passage).sents]
        sampled_passages = data[i]['t5_text_examples']
        # print(len(sentences))
        # print(len(sampled_passages))
        # print((sentences))
        # print((sampled_passages))
        mqag_score = selfcheck_mqag.predict(sentences = sentences, 
                                            passage = passage, 
                                            sampled_passages = sampled_passages, 
                                            num_questions_per_sent = 5, 
                                            scoring_method = 'bayes_with_alpha', 
                                            beta1 = 0.8, beta2 = 0.8)
        
        results['mqag'].append(mqag_score.tolist())
        print(mqag_score)
        # break
        
    return results

# batch_size = 30 
wikibio3_scores = find_SelfCheckGPT_score(gpt3_data, selfcheck_mqag)

with open("led_mqag_scores_next10.json", "w") as outfile: 
    json.dump(wikibio3_scores, outfile)