from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(BASE_DIR / ".cache" / "huggingface"))

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DATA_PATH = BASE_DIR / "data/sample_text.txt"
OUTPUT_DIR = BASE_DIR / "outputs"
SEQ_LEN = 40
WORD_SEQ_LEN = 8
EPOCHS = 1
BATCH_SIZE = 16
EMBED_DIM = 32
RNN_UNITS = 32


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    text = " ".join(text.split())
    return text.lower()


@dataclass
class TokenData:
    ids: list[int]
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    seq_len: int
    joiner: str


def char_tokenize(text: str) -> TokenData:
    tokens = sorted(set(text))
    token_to_id = {token: i for i, token in enumerate(tokens)}
    ids = [token_to_id[token] for token in text]
    id_to_token = {i: token for token, i in token_to_id.items()}
    return TokenData(ids, token_to_id, id_to_token, SEQ_LEN, "")


def word_tokenize(text: str) -> TokenData:
    words = text.split()
    tokens = sorted(set(words))
    token_to_id = {token: i for i, token in enumerate(tokens)}
    ids = [token_to_id[token] for token in words]
    id_to_token = {i: token for token, i in token_to_id.items()}
    return TokenData(ids, token_to_id, id_to_token, WORD_SEQ_LEN, " ")


class SimpleBPE:
    def __init__(self, vocab_size: int = 120) -> None:
        self.vocab_size = vocab_size
        self.merges: list[tuple[str, str]] = []

    def fit(self, words: list[str]) -> None:
        sequences = [tuple(list(word) + ["</w>"]) for word in words if word]
        vocab = set(token for seq in sequences for token in seq)

        # BPE склеивает самые частые пары соседних токенов.
        while len(vocab) < self.vocab_size:
            pairs: dict[tuple[str, str], int] = {}
            for seq in sequences:
                for pair in zip(seq, seq[1:]):
                    pairs[pair] = pairs.get(pair, 0) + 1

            if not pairs:
                break

            best_pair, best_count = max(pairs.items(), key=lambda item: item[1])
            if best_count < 2:
                break

            merged = "".join(best_pair)
            self.merges.append(best_pair)
            vocab.add(merged)
            sequences = [self._merge_pair(seq, best_pair, merged) for seq in sequences]

    @staticmethod
    def _merge_pair(seq: tuple[str, ...], pair: tuple[str, str], merged: str) -> tuple[str, ...]:
        result: list[str] = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                result.append(merged)
                i += 2
            else:
                result.append(seq[i])
                i += 1
        return tuple(result)

    def encode_word(self, word: str) -> list[str]:
        seq = tuple(list(word) + ["</w>"])
        for pair in self.merges:
            seq = self._merge_pair(seq, pair, "".join(pair))
        return [token for token in seq if token != "</w>"]

    def encode(self, text: str) -> list[str]:
        tokens: list[str] = []
        for word in text.split():
            tokens.extend(self.encode_word(word))
            tokens.append(" ")
        return tokens[:-1]


def bpe_tokenize(text: str) -> TokenData:
    tokenizer = SimpleBPE(vocab_size=120)
    tokenizer.fit(text.split())
    tokens = tokenizer.encode(text)
    vocab = sorted(set(tokens))
    token_to_id = {token: i for i, token in enumerate(vocab)}
    ids = [token_to_id[token] for token in tokens]
    id_to_token = {i: token for token, i in token_to_id.items()}
    return TokenData(ids, token_to_id, id_to_token, SEQ_LEN, "")


def make_dataset(ids: list[int], seq_len: int) -> TensorDataset:
    x_data: list[list[int]] = []
    y_data: list[int] = []

    for i in range(len(ids) - seq_len):
        x_data.append(ids[i : i + seq_len])
        y_data.append(ids[i + seq_len])

    x_tensor = torch.tensor(x_data, dtype=torch.long)
    y_tensor = torch.tensor(y_data, dtype=torch.long)
    return TensorDataset(x_tensor, y_tensor)


class TextGenerator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        architecture: str,
        layers: int = 1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)

        if architecture == "simple_rnn":
            self.rnn = nn.RNN(
                EMBED_DIM,
                RNN_UNITS,
                num_layers=layers,
                batch_first=True,
            )
            output_size = RNN_UNITS
        elif architecture == "lstm":
            self.rnn = nn.LSTM(
                EMBED_DIM,
                RNN_UNITS,
                num_layers=layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
            output_size = RNN_UNITS * 2 if bidirectional else RNN_UNITS
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        self.fc = nn.Linear(output_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        last_output = output[:, -1, :]
        return self.fc(last_output)


def build_model(vocab_size: int, architecture: str, layers: int = 1) -> TextGenerator:
    if architecture == "bilstm":
        return TextGenerator(vocab_size, "lstm", layers=1, bidirectional=True)
    return TextGenerator(vocab_size, architecture, layers=layers)


def train_epoch(
    model: TextGenerator,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * len(y_batch)
        total_correct += int((logits.argmax(dim=1) == y_batch).sum().item())
        total_items += len(y_batch)

    return total_loss / total_items, total_correct / total_items


def prepare_seed(data: TokenData) -> list[int]:
    if len(data.ids) <= data.seq_len:
        return data.ids.copy()
    return data.ids[: data.seq_len].copy()


def generate_text(
    model: TextGenerator,
    data: TokenData,
    device: torch.device,
    length: int = 120,
    temperature: float = 0.8,
) -> str:
    model.eval()
    generated = prepare_seed(data)

    with torch.no_grad():
        for _ in range(length):
            x = torch.tensor([generated[-data.seq_len :]], dtype=torch.long).to(device)
            logits = model(x)[0] / temperature
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            next_id = int(np.random.choice(len(probs), p=probs))
            generated.append(next_id)

    tokens = [data.id_to_token[token_id] for token_id in generated]
    return data.joiner.join(tokens)


def train_one(
    name: str,
    token_data: TokenData,
    architecture: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    layers: int = 1,
) -> dict[str, object]:
    dataset = make_dataset(token_data.ids, token_data.seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = build_model(len(token_data.token_to_id), architecture, layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    loss = 0.0
    accuracy = 0.0
    for epoch in range(epochs):
        loss, accuracy = train_epoch(model, loader, criterion, optimizer, device)
        print(f"epoch {epoch + 1}/{epochs} - loss: {loss:.4f} - accuracy: {accuracy:.4f}")

    sample = generate_text(model, token_data, device)

    return {
        "name": name,
        "architecture": architecture,
        "layers": layers,
        "vocab_size": len(token_data.token_to_id),
        "loss": float(loss),
        "accuracy": float(accuracy),
        "sample": sample,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение моделей для генерации текста.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Количество эпох для обучения.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Размер батча для обучения.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()
    OUTPUT_DIR.mkdir(exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    text = read_text(DATA_PATH)
    token_sets = {
        "char": char_tokenize(text),
        "word": word_tokenize(text),
        "bpe": bpe_tokenize(text),
    }

    experiments = [
        ("simple_rnn_char", "char", "simple_rnn", 1),
        ("simple_rnn_word", "word", "simple_rnn", 1),
        ("lstm_1_char", "char", "lstm", 1),
        ("lstm_2_char", "char", "lstm", 2),
        ("lstm_1_word", "word", "lstm", 1),
        ("lstm_2_word", "word", "lstm", 2),
        ("lstm_1_bpe", "bpe", "lstm", 1),
        ("lstm_2_bpe", "bpe", "lstm", 2),
        ("bilstm_char", "char", "bilstm", 1),
    ]

    results = []
    for name, token_name, architecture, layers in experiments:
        print(f"\nTraining {name}")
        result = train_one(
            name=name,
            token_data=token_sets[token_name],
            architecture=architecture,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            layers=layers,
        )
        results.append(result)

    result_path = OUTPUT_DIR / "results.json"
    result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nDone. Results are saved to outputs/results.json")


if __name__ == "__main__":
    main()
