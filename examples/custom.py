import csv
import random
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import gradiex

# ==========================================
# 1. PROGRAMMATIC DATASET GENERATOR (2000 Samples)
# ==========================================
THRESHOLD = 0.3
# Base word pools
explicit_vocab = [
    "fuck", "fucked", "fucking", "shit", "shitty", "bitch", "bitchy", 
    "ass", "asshole", "bastard", "dick", "pussy", "cock", "slut", 
    "motherfucker", "horny", "erotic", "nsfw", "porn", "naked", 
    "sex", "sexy", "nude", "orgasm", "crap", "damn"
]

safe_vocab = [
    "happy", "joy", "excited", "tired", "busy", "cool", "nice", 
    "awesome", "great", "friendly", "kind", "smart", "funny", 
    "calm", "focused", "eager", "polite", "honest", "helpful"
]

explicit_templates = [
    "you are a {word}",
    "this is completely {word}",
    "what the {word} is going on",
    "i feel so {word} right now",
    "get the {word} out of here",
    "that was a {word} move",
    "stop being such a {word}",
    "i am so {word} today",
    "look at this {word} picture",
    "send {word} images",
    "totally {word} up situation",
    "she is so {word}",
    "he is acting like a {word}",
    "just {word} off",
    "why are you being so {word}"
]

safe_templates = [
    "i am feeling very {word} today",
    "that was a {word} presentation",
    "she gave a {word} response",
    "we had a {word} conversation",
    "it is a {word} afternoon",
    "he is a really {word} person",
    "what a {word} idea",
    "this code works in a {word} way",
    "have a {word} time at the park",
    "learning python is super {word}",
    "let us have a {word} discussion",
    "the team was very {word} about the project",
    "everything is looking {word} so far",
    "i hope you have a {word} day",
    "this soup tastes quite {word}"
]

safe_phrases_and_nouns = [
    "hello world", "good morning", "how are you doing", "see you tomorrow",
    "thank you so much", "no problem at all", "let me check that for you",
    "hot chocolate", "hot coffee", "hot weather", "hot soup", "clue game",
    "oh no what happened", "bruh moment", "just chilling", "working hard",
    "machine learning model", "python script", "sentence embeddings", "vector database"
]

def generate_dataset():
    data = []
    
    # Generate 1000 Explicit (Label 1)
    for _ in range(1000):
        tmpl = random.choice(explicit_templates)
        word = random.choice(explicit_vocab)
        sentence = tmpl.format(word=word)
        data.append((sentence, 1))

    # Generate 1000 Safe (Label 0)
    for _ in range(1000):
        if random.random() < 0.3:
            sentence = random.choice(safe_phrases_and_nouns)
        else:
            tmpl = random.choice(safe_templates)
            word = random.choice(safe_vocab)
            sentence = tmpl.format(word=word)
        data.append((sentence, 0))

    random.shuffle(data)
    return data

# Generate and save to CSV
dataset = generate_dataset()
with open("explicit_dataset_2000.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["sentence", "label"])
    writer.writerows(dataset)

print(f"Successfully generated {len(dataset)} samples and saved to 'explicit_dataset_2000.csv'.")

# ==========================================
# 2. EMBEDDING & GRADIEX TRAINING
# ==========================================

# Load generated dataset
df = pd.read_csv("explicit_dataset_2000.csv")
sentences = df["sentence"].tolist()
labels = df["label"].values.reshape(-1, 1)

print("Encoding 2,000 sentences into vector embeddings...")
_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = _model.encode(sentences, show_progress_bar=True)
embedding_dim = embeddings.shape[1]  # 384

# Setup Gradiex Model
gradiex.seed(42)
net = gradiex.Network(learning_rate=0.01, batch_size=32)  # Increased batch size for 2k samples
net.add_hidden(16, input_size=embedding_dim, activation="leaky_relu", init="xavier")
net.add_output(1, activation="sigmoid", loss="binary_cross_entropy")
net.build()

print("Training gradiex network...")
history = net.train(embeddings, labels, epochs=300, shuffle=True, seed=42, plot=True)

def save(fig, name, theme="light"):
    path = f"./{name}"
    fig.savefig(path, dpi=130, facecolor=gradiex.THEMES[theme]["surface"])
    return path

save(net.plot_loss(title="2k Dataset Training Loss"), "02_loss_2k.png")

# ==========================================
# 3. INFERENCE LOOP
while True:
    inp = input("\nSay Something (or press Enter to exit): ")
    if not inp.strip():
        break
    
    # Encode and flatten to 1D vector of shape (384,)
    matrix = _model.encode(inp).flatten()
    
    # Predict directly
    pred = net.predict(matrix)
    
    # If gradiex returns a single float/scalar or 1D array:
    score = float(pred[0]) if isinstance(pred, (np.ndarray, list)) else float(pred)

    label = "Explicit" if score > THRESHOLD else "Not Explicit"
    print(f"Result: {label} (Score: {score:.4f})")