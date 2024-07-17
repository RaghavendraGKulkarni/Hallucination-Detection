import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import json
import threading
from queue import Queue
import torch
from transformers import pipeline, BartForConditionalGeneration, AutoTokenizer
import re
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

with open('gpt3_wikibio.json', 'r') as file:
    dataset = json.load(file)

print("size of the dataset is ", len(dataset))

model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(device)
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')


def clean_summary(summary):
    pattern = re.compile(r'\[\w+\]')
    return re.sub(pattern, '', summary).strip()


def process_batch(batch, model, tokenizer, device):
    batch_json_strings = []
    bios = [row['wiki_bio_text'] for row in batch]
    inputs = tokenizer(bios, max_length=1024, return_tensors='pt', padding=True, truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    for idx, row in tqdm(enumerate(batch)):
        main_summary_ids = model.generate(inputs['input_ids'][idx].unsqueeze(0), num_beams=4, length_penalty=2.0, max_length=150, min_length=50, no_repeat_ngram_size=3)
        main_summary = clean_summary(tokenizer.decode(main_summary_ids[0], skip_special_tokens=True))
        samples = []
        for _ in range(5):
            short_summary_ids = model.generate(inputs['input_ids'][idx].unsqueeze(0), num_beams=4, length_penalty=0.5, max_length=150, min_length=50, no_repeat_ngram_size=3, do_sample=True, top_k=50, temperature=1.2)
            short_summary = clean_summary(tokenizer.decode(short_summary_ids[0], skip_special_tokens=True))
            samples.append(short_summary)

        json_data = {
            'bert_text': main_summary,
            'bert_text_examples': samples,
            'wiki_bio_text': row['wiki_bio_text']
        }
        batch_json_strings.append(json_data)

    return batch_json_strings


batch_size = 1  
batches = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]

json_strings = []
for batch in batches:
    json_strings.extend(process_batch(batch, model, tokenizer, device))

output_file_path = 'bart_base_dataset.json'
with open(output_file_path, 'w') as f:
    json.dump(json_strings, f, indent=4)

print(f"Cleaned JSON output has been written to: {output_file_path}")