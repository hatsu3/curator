import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import VerificationMode, load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from benchmark.config import DatasetConfig
from dataset import DATA_DIR

ARXIV_DATA_DIR = Path(DATA_DIR) / "arxiv"
ARXIV_DATA_DIR.mkdir(parents=True, exist_ok=True)


class SentenceEmbedder:
    def __init__(
        self, model_name="sentence-transformers/all-MiniLM-L6-v2", use_cuda=True
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.device = "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.to(self.device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def embed_sentences(self, sentences):
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = self._mean_pooling(
            model_output, encoded_input["attention_mask"]
        )
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


def preprocess_arxiv_dataset(
    num_categories=100,
    sample_size=10000,
    seed=42,
    data_dir=ARXIV_DATA_DIR,
    output_path=ARXIV_DATA_DIR / "processed.pkl",
):
    random.seed(seed)

    data_dir = Path(data_dir)
    dataset = load_dataset(
        "arxiv_dataset",
        data_dir=str(data_dir),
        verification_mode=VerificationMode.NO_CHECKS,
    )
    dataset_len = len(dataset["train"])  # type: ignore

    # find the most common categories
    subset_indices = random.sample(
        range(dataset_len), min(dataset_len, sample_size * 10)
    )
    subset = [dataset["train"][i] for i in subset_indices]  # type: ignore
    categories = [entry["categories"] for entry in subset]

    category_count = {}
    for category in categories:
        for c in category.split():
            if c not in category_count:
                category_count[c] = 0
            category_count[c] += 1

    top_categories = sorted(category_count.items(), key=lambda x: x[1], reverse=True)[
        :num_categories
    ]
    top_categories = [x[0] for x in top_categories]

    # pick the first sample_size entries that have at least one of the top categories
    filtered_entries = []
    for entry in tqdm(dataset["train"], total=dataset_len):  # type: ignore
        entry_cates = entry["categories"].split()  # type: ignore
        entry_cates = [cate for cate in entry_cates if cate in top_categories]

        if len(entry_cates) == 0:
            continue

        filtered_entries.append(
            {
                "id": entry["id"],  # type: ignore
                "categories": entry_cates,  # type: ignore
                "abstract": entry["abstract"],  # type: ignore
            }
        )

        if len(filtered_entries) >= sample_size:
            categories_in_sample = set()
            for entry in filtered_entries:
                categories_in_sample.update(entry["categories"])
            print("categories in sample:", len(categories_in_sample))
            break

    # store the processed dataset in another pickle file
    with open(output_path, "wb") as f:
        print("Saving processed dataset to", output_path)
        pickle.dump(filtered_entries, f)


def gen_arxiv_embeddings(
    batch_size=8,
    pkl_path=ARXIV_DATA_DIR / "processed.pkl",
    embed_path=ARXIV_DATA_DIR / "embeddings.npy",
):
    if not Path(pkl_path).exists():
        preprocess_arxiv_dataset()

    with open(pkl_path, "rb") as f:
        dataset = pickle.load(f)

    embedder = SentenceEmbedder()
    embeddings = []
    for i in tqdm(range(0, len(dataset), batch_size)):
        sentences = [entry["abstract"] for entry in dataset[i : i + batch_size]]
        embeddings.append(embedder.embed_sentences(sentences))

    embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
    np.save(embed_path, embeddings)


def load_arxiv_dataset_vecs(
    embed_path=ARXIV_DATA_DIR / "embeddings.npy",
    test_size=0.2,
    seed=42,
):
    embeddings = np.load(embed_path)

    # split into train and test
    dataset_size = len(embeddings)
    train_size = int(dataset_size * (1 - test_size))
    np.random.seed(seed=seed)

    shuffled_idxs = np.random.permutation(dataset_size)
    train_indices = shuffled_idxs[:train_size]
    test_indices = shuffled_idxs[train_size:]

    train_vecs = embeddings[train_indices]
    test_vecs = embeddings[test_indices]

    return train_vecs, test_vecs


def load_arxiv_dataset_mds(
    pkl_path=ARXIV_DATA_DIR / "processed.pkl",
    share_degree: int | None = None,
    test_size=0.2,
    seed=42,
):
    with open(pkl_path, "rb") as f:
        dataset = pickle.load(f)

    categories = set()
    for entry in dataset:
        categories.update(entry["categories"])
    categories = sorted(categories)
    category2id = {category: i for i, category in enumerate(categories)}

    # generate access list for each vector
    owners = [[category2id[cate] for cate in entry["categories"]] for entry in dataset]

    if share_degree is not None:
        share_deg_before = np.mean([len(owner) for owner in owners])
        expand_factor = round(share_degree / share_deg_before)

        if expand_factor <= 1:
            print(
                "Target share degree is smaller than the original one. No need to expand."
            )
        else:
            np.random.seed(seed=seed)
            n_cates = len(categories)
            cate_id_to_users = [
                np.random.choice(n_cates, expand_factor).tolist()
                for _ in range(n_cates)
            ]
            owners = [
                sum([cate_id_to_users[cate_id] for cate_id in owner], [])
                for owner in owners
            ]
            share_deg_after = np.mean([len(owner) for owner in owners])

            print("Share degree before expansion:", share_deg_before)
            print("Share degree after expansion:", share_deg_after)

    # split into train and test
    dataset_size = len(owners)
    train_size = int(dataset_size * (1 - test_size))
    np.random.seed(seed=seed)
    shuffled_idxs = np.random.permutation(dataset_size)
    train_indices = shuffled_idxs[:train_size]
    test_indices = shuffled_idxs[train_size:]

    train_mds = [owners[idx] for idx in train_indices]
    test_mds = [owners[idx] for idx in test_indices]

    # randomly generate test metadata
    n_cates = len(categories)
    test_mds = [
        np.random.choice(n_cates, size=len(mds), replace=False).tolist()
        for mds in test_mds
    ]

    return train_mds, test_mds


@dataclass
class ArxivDatasetConfig(DatasetConfig):
    dataset_name: str = field(default="arxiv", init=False, repr=True)

    def validate_params(self):
        assert self.dataset_params.keys() <= {
            "pkl_path",
            "embed_path",
            "test_size",
            "seed",
        }
        self._validate_metadata_params()


if __name__ == "__main__":
    suffix = "user100_vec2e6"
    pkl_path = ARXIV_DATA_DIR / f"processed_{suffix}.pkl"
    embed_path = ARXIV_DATA_DIR / f"embeddings_{suffix}.npy"
    filtered_entries = preprocess_arxiv_dataset(
        num_categories=100, sample_size=2000000, output_path=pkl_path
    )
    gen_arxiv_embeddings(
        batch_size=32,
        pkl_path=pkl_path,
        embed_path=embed_path,
    )
    train_vecs, test_vecs = load_arxiv_dataset_vecs(embed_path=embed_path)
    train_mds, test_mds = load_arxiv_dataset_mds(pkl_path=pkl_path)
