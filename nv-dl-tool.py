import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from scipy.stats import sem
import pandas as pd
import os
import time
from tqdm import tqdm
from datetime import datetime
from torch.cuda.amp import GradScaler, autocast  # For mixed precision training
import csv
from collections import Counter
from transformers import logging
logging.set_verbosity_error()

class TDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(preds, labels):
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    f1_micro = f1_score(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    std_dev = sem(preds)
    return {
        'accuracy': acc,
        'f1_weighted': f1,
        'f1_micro': f1_micro,
        'precision': precision,
        'recall': recall,
        'std_dev': std_dev
    }

def train_class(df,class_i, model_name='roberta-base', epochs=4, batch_size=64, learning_rate=1e-5, patience=1):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    le = LabelEncoder()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    os.makedirs('tuned_models', exist_ok=True)

    texts = df['description'].tolist()
    labels = df[class_i].tolist()

    train_texts, temp_test_texts, train_labels, temp_test_labels = train_test_split(texts, labels, test_size=0.15, random_state=42)
    val_texts, test_texts, val_labels, test_labels = train_test_split(temp_test_texts, temp_test_labels, test_size=0.5, random_state=42)

    le.fit(train_labels + val_labels + test_labels)
    train_labels_encodings = le.transform(train_labels)
    val_labels_encodings = le.transform(val_labels)
    test_labels_encodings = le.transform(test_labels)
    # Create the label mapping
    label_mapping = {index: class_ for index, class_ in enumerate(le.classes_)}

    class_weights = [len(labels) / Counter(labels)[class_] for class_ in le.classes_]
    class_weights = torch.tensor(class_weights).to(device)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=64)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=64)

    train_dataset = TDataset(train_encodings, train_labels_encodings)
    val_dataset = TDataset(val_encodings, val_labels_encodings)
    test_dataset = TDataset(test_encodings, test_labels_encodings)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(set(train_labels_encodings)))
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
    scaler = GradScaler()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*epochs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_accuracy = 0
    epochs_no_improve = 0
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Class: {class_i} Epoch: {epoch+1}', dynamic_ncols=True)
        model.train()
        for i, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            with autocast():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
                
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': '{:.3f}'.format(total_loss/(progress_bar.n+1))}, refresh=True)

        model.eval()
        val_preds, val_labels = [], []
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(device)
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
            preds_batch = outputs.logits.argmax(dim=-1).cpu().numpy()
            labels_batch = labels_batch.cpu().numpy()
            val_preds.extend(preds_batch)
            val_labels.extend(labels_batch)

        val_metrics = compute_metrics(val_preds, val_labels)
        val_accuracy = val_metrics['accuracy']
        print(f"\nEpoch: {epoch+1}, Loss: {total_loss/len(train_loader)}, Validation Accuracy: {val_accuracy}, Learning Rate: {scheduler.get_last_lr()[0]}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            model.save_pretrained(os.path.join('tuned_models', f'{class_i}_model'))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early stopping due to no improvement in accuracy.")
                break

    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    test_preds, test_labels = [], []
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_batch = batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        preds_batch = outputs.logits.argmax(dim=-1).cpu().numpy()
        labels_batch = labels_batch.cpu().numpy()
        test_preds.extend(preds_batch)
        test_labels.extend(labels_batch)

    test_metrics = compute_metrics(test_preds, test_labels)
    print(f"Final test metrics for {class_i}: {test_metrics}")

    # Append metrics to CSV
    metrics_file = 'metrics.csv'
    file_exists = os.path.isfile(metrics_file)
    with open(metrics_file, 'a') as f:
        writer = csv.DictWriter(f, fieldnames=['class', 'timestamp', 'accuracy', 'f1_weighted', 'f1_micro', 'precision', 'recall', 'std_dev'])
        if not file_exists:
            writer.writeheader()
        writer.writerow({**{'class': class_i, 'timestamp': datetime.now()}, **test_metrics})

    # Append label mapping to CSV
    label_mapping_file = 'label_mapping.csv'
    file_exists = os.path.isfile(label_mapping_file)
    with open(label_mapping_file, 'a') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['encoded_label', 'actual_label'])
        for key, value in label_mapping.items():
            writer.writerow([key, value])



df = pd.read_csv('nlp-pp-db.csv')
classes = df.columns.to_list()
classes.remove('description')

model_name = 'roberta-base' #could be changed with any language models :)
epochs = 15
batch_size = 128
learning_rate = 0.00005
classes.reverse()
classes_to_train = classes
for class_i in classes_to_train:
    print(f"Training Class: {class_i}")
    train_class(df,class_i, model_name=model_name, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    print(f"Finished training {class_i}")

print("Training of all classes is complete.")
