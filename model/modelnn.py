import nltk
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split  # type: ignore

# Download utility for reading CSV files
#nltk.download('punkt')

# === Dataset class for data processing ===
class QuestionAnswerDataset(Dataset):
    def __init__(self, dataframe, vocab, max_len=None):
        # Tokenize everything
        if 'tokenize_question' not in dataframe.columns or 'tokenize_answer' not in dataframe.columns:
            raise ValueError("DataFrame must contain 'tokenize_question' and 'tokenize_answer' columns.")
        if "<UNK>" not in vocab:
            raise ValueError("Vocabulary must contain a '<UNK>' token.")
        self.data = dataframe
        self.vocab = vocab
        self.max_len = max_len

    # Get the number of samples in the dataset
    def __len__(self):
        return len(self.data)

    # Split the data into tokens
    def __getitem__(self, idx):
        question_tokens = self.data.iloc[idx]['tokenize_question'].split()
        answer_tokens = self.data.iloc[idx]['tokenize_answer'].split()

        question_indices = [self.vocab.get(token, self.vocab["<UNK>"]) for token in question_tokens]
        answer_indices = [self.vocab.get(token, self.vocab["<UNK>"]) for token in answer_tokens]

        if self.max_len:
            question_indices = self.pad_sequence(question_indices, self.max_len)
            answer_indices = self.pad_sequence(answer_indices, self.max_len)

        return torch.tensor(question_indices, dtype=torch.long), torch.tensor(answer_indices, dtype=torch.long)

    def pad_sequence(self, sequence, max_len, pad_token="<PAD>"):
        return sequence + [self.vocab.get(pad_token, 0)] * (max_len - len(sequence))

# === Build a vocabulary from tokens ===
def build_vocab(dataframe):
    tokens = set()
    for col in ['tokenize_question', 'tokenize_answer']:
        for row in dataframe[col]:
            tokens.update(row.split())
    vocab = {token: idx for idx, token in enumerate(tokens, start=1)}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = len(vocab)  # Fix for missing token
    return vocab

def collate_fn(batch):
    questions, answers = zip(*batch)

    # Padding to align sequences
    questions_padded = pad_sequence(questions, batch_first=True, padding_value=vocab["<PAD>"])
    answers_padded = pad_sequence(answers, batch_first=True, padding_value=vocab["<PAD>"])

    return questions_padded, answers_padded

# === Seq2Seq model ===
class PrototypeModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout_rate=0.30):
        super(PrototypeModel, self).__init__()
        # Convert to vectors, encode, decode, and predict answers based on encoded data

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, question, answer):
        # Embedding
        embedded_question = self.embedding(question)
        embedded_answer = self.embedding(answer)

        embedded_question = self.dropout(embedded_question)

        # Encoder
        _, (hidden, cell) = self.encoder(embedded_question)

        embedded_answer = self.dropout(embedded_answer)

        # Decoder
        output, _ = self.decoder(embedded_answer, (hidden, cell))

        # Logits for word prediction
        output = self.fc(output)
        return output

# === Load data ===
df = pd.read_csv('dataset/dataset1.csv')

# Build token vocabulary
vocab = build_vocab(df)

# Create Dataset and DataLoader
dataset = QuestionAnswerDataset(df, vocab)
data_loader = DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=collate_fn)

# === Initialize model ===
vocab_size = len(vocab)
embed_size = 50
hidden_size = 100

model = PrototypeModel(vocab_size, embed_size, hidden_size)

# === Loss function and optimizer ===
criterion = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === Split data ===
train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Create datasets
train_dataset = QuestionAnswerDataset(train_data, vocab)
val_dataset = QuestionAnswerDataset(val_data, vocab)
test_dataset = QuestionAnswerDataset(test_data, vocab)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False, collate_fn=collate_fn)

# === Training loop ===
epochs = 100

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for question, answer in train_loader:
        optimizer.zero_grad()

        answer_input = answer[:, :-1]
        answer_target = answer[:, 1:]

        output = model(question, answer_input)
        output = output.reshape(-1, vocab_size)
        answer_target = answer_target.reshape(-1)

        loss = criterion(output, answer_target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Validation after each epoch
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for question, answer in val_loader:
            answer_input = answer[:, :-1]
            answer_target = answer[:, 1:]

            output = model(question, answer_input)
            output = output.reshape(-1, vocab_size)
            answer_target = answer_target.reshape(-1)

            loss = criterion(output, answer_target)
            val_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Validation Loss: {val_loss:.4f}")

# === Evaluate the model ===
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for question, answer in test_loader:
            answer_input = answer[:, :-1]
            answer_target = answer[:, 1:]

            output = model(question, answer_input)
            output = output.reshape(-1, vocab_size)
            answer_target = answer_target.reshape(-1)

            loss = criterion(output, answer_target)
            test_loss += loss.item()

    print(f"Test Loss: {test_loss:.4f}")
    return test_loss

# Call after training
evaluate_model(model, test_loader, criterion)

# === Save the model ===
torch.save(model.state_dict(), 'model\prototype003.pth')
print("Model saved!")

# === Prediction function ===
def predict_answer(model, question, vocab, max_len=20):
    model.eval()
    tokens = word_tokenize(question.lower())
    question_indices = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    question_tensor = torch.tensor([question_indices], dtype=torch.long)

    with torch.no_grad():
        embedded_question = model.embedding(question_tensor)
        _, (hidden, cell) = model.encoder(embedded_question)

        answer_indices = []
        next_token = vocab["<PAD>"]

        for _ in range(max_len):
            input_tensor = torch.tensor([[next_token]], dtype=torch.long)
            embedded_answer = model.embedding(input_tensor)
            output, (hidden, cell) = model.decoder(embedded_answer, (hidden, cell))
            logits = model.fc(output.squeeze(0))
            next_token = logits.argmax(dim=-1).item()
            if next_token == vocab["<PAD>"]:
                break
            answer_indices.append(next_token)

    idx_to_token = {idx: token for token, idx in vocab.items()}
    answer_tokens = [idx_to_token.get(idx, "<UNK>") for idx in answer_indices]
    return " ".join(answer_tokens)
