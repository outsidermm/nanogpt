import torch

from data import CharTokenizer, get_batch, load_data


def test_tokenizer_roundtrip():
    tokenizer = CharTokenizer(sorted(set("hello world")))
    text = "hello world"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_tokenizer_vocab_size():
    tokenizer = CharTokenizer(["a", "b", "c"])
    assert tokenizer.vocab_size == 3


def test_load_data_splits_and_vocab(tmp_path):
    text = "abcabcabcabc"
    path = tmp_path / "corpus.txt"
    path.write_text(text)

    train_data, val_data, tokenizer = load_data(str(path), train_split=0.5)

    assert len(train_data) + len(val_data) == len(text)
    assert tokenizer.vocab_size == 3  # a, b, c
    assert tokenizer.decode(train_data.tolist() + val_data.tolist()) == text


def test_get_batch_shapes_and_targets_are_shifted(tmp_path):
    text = "abcdefghij" * 5
    path = tmp_path / "corpus.txt"
    path.write_text(text)
    data, _, _tokenizer = load_data(str(path), train_split=1.0)

    context_size, batch_size = 4, 8
    x, y = get_batch(data, context_size, batch_size, device="cpu")

    assert x.shape == (batch_size, context_size)
    assert y.shape == (batch_size, context_size)
    # y is x shifted one position to the right within the source sequence
    assert torch.equal(x[:, 1:], y[:, :-1])
