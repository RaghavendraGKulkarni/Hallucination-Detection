import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from transformers import AutoTokenizer, FlaxT5ForConditionalGeneration
import jax
from tqdm import tqdm
import random
import json
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Read dataset
with open('gpt3_wikibio.json', 'r') as file:
    dataset = json.load(file)[:100]  # Only take the first 100 entries

print("Size of the dataset is", len(dataset))

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = FlaxT5ForConditionalGeneration.from_pretrained("t5-small")

def clean_summary(summary):
    """Clean special tokens and unwanted spaces from the summary."""
    return summary.strip()


def process_batch(batch, model, tokenizer):
    batch_json_strings = []
    bios = ['summarize: ' + row['wiki_bio_text'] for row in batch]
    inputs = tokenizer(bios, return_tensors='jax', padding=True, truncation=True, max_length=1024)

    for idx, bio in enumerate(bios):
        # Generate the main summary
        main_summary_ids = model.generate(inputs['input_ids'][idx:idx+1], num_beams=4, max_length=150, min_length=50, no_repeat_ngram_size=3)
        main_summary = clean_summary(tokenizer.decode(main_summary_ids.sequences[0], skip_special_tokens=True))

        # Generate twenty short summaries with variations
        short_summaries = []
        for _ in range(5):
            # Randomize parameters for variation
            length_penalty = random.uniform(0.1, 1.0)  # Shorter or longer summaries
            # num_beams = random.randint(1, 4)  # Number of beams for beam search
            temperature = random.uniform(0.9, 1.3)  # Sampling temperature
            top_k = random.randint(20, 50)  # Top-k sampling
            top_p = random.uniform(0.7, 1.0)  # Top-p (nucleus) sampling

            short_summary_ids = model.generate(
                inputs['input_ids'][idx:idx+1],
                length_penalty=length_penalty,
                max_length=150,
                min_length=50,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature
            )
            short_summary = clean_summary(tokenizer.decode(short_summary_ids.sequences[0], skip_special_tokens=True))
            short_summaries.append(short_summary)

        # Append the data to the batch_json_strings list
        json_data = {
            't5_text': main_summary,
            't5_text_examples': short_summaries,
            'wiki_bio_text': batch[idx]['wiki_bio_text']
        }
        batch_json_strings.append(json_data)
        
        if idx==101:
            break

    return batch_json_strings






# Define batch size and create batches
batch_size = 1
batches = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]

json_strings = []
# Process batches and generate summaries
for batch in tqdm(batches, desc="Generating summaries"):
    json_strings.extend(process_batch(batch, model, tokenizer))

# Save the generated summaries to a file
output_file_path = 't5_dataset.json'
with open(output_file_path, 'w') as f:
    json.dump(json_strings, f, indent=4)

print(f"Summaries have been written to: {output_file_path}")