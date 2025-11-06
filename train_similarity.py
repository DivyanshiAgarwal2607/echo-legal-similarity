from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import csv
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MODEL_BASE = os.getenv("MODEL_NAME", "all-mpnet-base-v2")

def load_pairs(csv_path):
    """
    Load training pairs (anchor, positive) from a CSV file.
    Format: anchor,positive
    """
    examples = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            anchor = row['anchor'].strip()
            positive = row['positive'].strip()
            if anchor and positive:
                examples.append(InputExample(texts=[anchor, positive]))
    return examples

def train(csv_path, out_dir="models/legal-sim-model", epochs=2, batch_size=8):
    """
    Train a contrastive model using MultipleNegativesRankingLoss.
    """
    model = SentenceTransformer(MODEL_BASE)
    train_examples = load_pairs(csv_path)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    print(f"🚀 Training model on {len(train_examples)} pairs...")
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=epochs, warmup_steps=100)
    model.save(out_dir)
    print(f"✅ Model saved to {out_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train similarity model on case pairs.")
    parser.add_argument("--csv", required=True, help="Path to training CSV file (with anchor,positive columns)")
    parser.add_argument("--out", default="models/legal-sim-model", help="Output folder for trained model")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    train(args.csv, out_dir=args.out, epochs=args.epochs, batch_size=args.batch)
