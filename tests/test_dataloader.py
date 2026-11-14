"""Regression tests for how the pretraining packer treats documents longer than a row."""

import itertools

import nanochat.dataloader as dataloader


class TokenizedDocuments:
    """Documents are explicit token lists, so packing choices are observable."""

    def get_bos_token_id(self):
        return 0

    def encode(self, documents, prepend, num_threads):
        return [[prepend, *document] for document in documents]


LONG = list(range(100, 120))
SHORT = [1, 2, 3]
MEDIUM = [4, 5, 6, 7]


def first_row(monkeypatch, **options):
    batches = itertools.repeat(([LONG, SHORT, MEDIUM], (0, 0, 1)))
    monkeypatch.setattr(dataloader, "_document_batches", lambda *args: batches)
    loader = dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(
        TokenizedDocuments(), 1, 8, "train", device="cpu", buffer_size=2, **options,
    )
    inputs, targets, _ = next(loader)
    return inputs[0].tolist(), targets[0].tolist()


def test_document_longer_than_a_row_becomes_one_whole_row(monkeypatch):
    inputs, targets = first_row(monkeypatch)
    assert inputs == [0, *LONG[:7]]
    assert targets == LONG[:8]


def test_uncropped_long_document_waits_while_shorter_documents_fit(monkeypatch):
    inputs, targets = first_row(monkeypatch, crop_long_documents=False)
    assert inputs == [0, *MEDIUM, 0, *SHORT[:2]]
    assert targets == [*MEDIUM, 0, *SHORT]
