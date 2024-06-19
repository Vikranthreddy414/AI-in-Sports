import pandas as pd

# Read the CSV file
df = pd.read_csv('data.csv')

# Function to combine topics and keywords into a single text string
def combine_text(row):
    topics = row['topics'].strip("[]").replace("'", "").split(", ")
    keywords = row['keywords'].strip("[]").replace("'", "").split(", ")
    
    # Convert topics list to a single string
    topic_text = " ".join(topics) if topics else "Topic not specified."
    
    # Convert keywords list to a single string
    keyword_text = ", ".join(keywords)
    
    return f"Topic: {topic_text}. Keywords: {keyword_text}."

# Apply function to create combined_text column
df['combined_text'] = df.apply(combine_text, axis=1)

# Save the resulting DataFrame to a new CSV file
df.to_csv('combined_data.csv', index=False)

# Display DataFrame
print(df)


# Import necessary libraries
from sentence_transformers import SentenceTransformer
from transformers import pipeline, Trainer, TrainingArguments, BertTokenizer, BertForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
from datasets import load_dataset
import ast
import re

# Function for basic text cleaning
def clean_text(text):
    # Remove leading/trailing spaces and convert to lowercase
    text = text.strip().lower()
    # Remove extraneous characters if any (example: quotes, brackets)
    text = re.sub(r"[\"\[\]]", "", text)
    return text

# Step 1: Load Labeled Data
df = pd.read_csv('labeled_data.csv')  # Replace 'labeled_data.csv' with the path to your file

# Preprocess data: Convert string representations of lists to actual lists and clean text
df['text'] = df['Topic'].apply(lambda x: clean_text(ast.literal_eval(x)[0]))  # Convert to string and clean
df['keywords'] = df['keywords'].apply(lambda x: [clean_text(keyword) for keyword in ast.literal_eval(x)])  # Convert to list and clean
df['label'] = df['label'].astype(str).apply(clean_text)  # Ensure labels are strings and clean

# Extract data
text_data = df['text'].tolist()
keywords_data = df['keywords'].tolist()
labels_data = df['label'].tolist()

# Extract unique domain labels
domain_labels = list(set(labels_data))

# Step 2: Load Models
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
zero_shot_classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

# Step 3: Create Embeddings for Domain Labels
domain_label_embeddings = sentence_model.encode(domain_labels)

# Step 4: Fine-tune a BERT model on your labeled data
# Prepare dataset for fine-tuning
dataset = pd.DataFrame({'text': text_data, 'label': labels_data})
dataset.to_csv('fine_tuning_data.csv', index=False)

# Load dataset using Hugging Face's datasets library
dataset = load_dataset('csv', data_files='fine_tuning_data.csv')

# Load tokenizer and BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(domain_labels))

# Tokenize data
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Map labels to indices
label2id = {label: idx for idx, label in enumerate(domain_labels)}
id2label = {idx: label for label, idx in label2id.items()}

def encode_labels(examples):
    examples['label'] = [label2id[label] for label in examples['label']]
    return examples

tokenized_datasets = tokenized_datasets.map(encode_labels, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
)

# Train model
trainer.train()

# Step 5: Zero-Shot Classification
def zero_shot_classify(topic, candidate_labels):
    result = zero_shot_classifier(topic, candidate_labels)
    return result['labels'][0]  # Top prediction

# Step 6: Compute Similarity Scores
def sentence_transformer_classify(topic, keywords, domain_label_embeddings, domain_labels):
    # Encode topic using Sentence Transformers
    topic_embedding = sentence_model.encode(topic)
    avg_keyword_embedding = np.mean(sentence_model.encode(keywords), axis=0)
    combined_embedding = 0.5 * (topic_embedding + avg_keyword_embedding)
    
    # Compute similarity with domain labels
    cosine_similarities = 1 - cdist([combined_embedding], domain_label_embeddings, 'cosine')
    return domain_labels[np.argmax(cosine_similarities)]

# Step 7: Combine Results
def classify_topic(topic, keywords, zero_shot_classifier, sentence_model, domain_label_embeddings, domain_labels, model, tokenizer):
    zero_shot_label = zero_shot_classify(topic, domain_labels)
    st_label = sentence_transformer_classify(topic, keywords, domain_label_embeddings, domain_labels)
    
    # Tokenize and classify using fine-tuned BERT
    inputs = tokenizer(topic, return_tensors='pt')
    outputs = model(**inputs)
    bert_prediction = outputs.logits.argmax(-1).item()
    bert_label = domain_labels[bert_prediction]
    
    # Weighted combination of predictions
    predictions = {
        'zero_shot': zero_shot_label,
        'sentence_transformer': st_label,
        'bert': bert_label
    }
    
    # Simple heuristic: prioritize Sentence Transformer and BERT results
    if st_label == bert_label:
        return st_label
    else:
        # Implement a more complex heuristic or weighted voting if needed
        return bert_label

# Classify each topic
for topic, kw in zip(text_data, keywords_data):
    combined_label = classify_topic(topic, kw, zero_shot_classifier, sentence_model, domain_label_embeddings, domain_labels, model, tokenizer)
    print(f"Topic: {topic}\nCombined Label: {combined_label}\n")

# Step 8: Implement Active Learning (simplified example)
def active_learning_feedback(topic, actual_label, model, tokenizer, domain_labels):
    # Encode the new topic
    inputs = tokenizer(topic, return_tensors='pt')
    actual_label_id = domain_labels.index(actual_label)
    
    # Add the new example to the training data
    new_example = {'text': topic, 'label': actual_label_id}
    tokenized_datasets['train'] = tokenized_datasets['train'].add_item(new_example)
    
    # Retrain the model with the new data
    trainer.train()

# Simulate feedback loop
new_topic = "Issue with tax filings and IRS submissions"
actual_label = "Tax"
active_learning_feedback(new_topic, actual_label, model, tokenizer, domain_labels)
