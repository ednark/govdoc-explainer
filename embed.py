from transformers import BertModel, BertTokenizer
import torch

# Load pre-trained model and tokenizer
embed_model_name = "bert-base-uncased"
embed_model = BertModel.from_pretrained(embed_model_name)
embed_tokenizer = BertTokenizer.from_pretrained(embed_model_name)

def generate_embeddings(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # The embeddings are the output of the last hidden state
    embeddings = outputs.last_hidden_state
    return embeddings.mean(dim=1).squeeze().numpy()

# Function to split text into chunks
def split_text_into_overlapping_chunks(text, max_length, overlap=0):
    words = text.split()
    chunks = []
    step = max_length - overlap
    for i in range(0, len(words), step):
        chunk = words[i:i + max_length]
        chunks.append(' '.join(chunk))
        if len(chunk) < max_length:
            break
    return chunks

text = ""
max_tokens = 500
token_overlap_window = 250
with open(text_file, "r") as file:
    text = file.read()

chunks = split_text_into_overlapping_chunks(text, max_tokens, overlap=token_overlap_window)

embeddings = []
for chunk in chunks:
    inputs = tokenizer(chunk, return_tensors='pt', truncation=True, padding='max_length', max_length=max_tokens)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings.append(outputs.last_hidden_state.mean(dim=1))

# Concatenate embeddings
final_embedding = torch.cat(embeddings, dim=1)

torch.save(final_embedding, embed_file)
print(f"Embedding saved to {embed_file}")
