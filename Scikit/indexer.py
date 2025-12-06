import json
import os
from pathlib import Path
from typing import List

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle


class Indexer:
    def __init__(self, docs_jsonl_path: str):
        self.docs_jsonl_path = docs_jsonl_path
        self.docs = []  # list of dicts with id,url,text
        self.id_to_meta = {}
        self._load_documents()

        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None

    def _load_documents(self):
        if not os.path.exists(self.docs_jsonl_path):
            raise FileNotFoundError(self.docs_jsonl_path)
        with open(self.docs_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                self.docs.append(doc)
                self.id_to_meta[doc['id']] = {'url': doc.get('url'), 'depth': doc.get('depth')}

    def build(self, save_dir: str = '.'):
        texts = [d.get('text', '') for d in self.docs]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

        # Build inverted index: term -> list of (doc_id, tfidf)
        terms = self.vectorizer.get_feature_names_out()
        inverted = {}
        # iterate over non-zero elements
        coo = self.tfidf_matrix.tocoo()
        for term_idx, doc_idx, value in zip(coo.col, coo.row, coo.data):
            term = terms[term_idx]
            doc_id = self.docs[doc_idx]['id']
            inverted.setdefault(term, []).append({'doc_id': doc_id, 'score': float(value)})

        os.makedirs(save_dir, exist_ok=True)
        # Save inverted index
        with open(Path(save_dir) / 'inverted_index.json', 'w', encoding='utf-8') as f:
            json.dump(inverted, f, ensure_ascii=False, indent=2)

        # Save metadata
        with open(Path(save_dir) / 'doc_meta.json', 'w', encoding='utf-8') as f:
            json.dump(self.id_to_meta, f, ensure_ascii=False, indent=2)

        # Save document order (matrix row -> doc_id)
        doc_order = [d['id'] for d in self.docs]
        with open(Path(save_dir) / 'doc_order.json', 'w', encoding='utf-8') as f:
            json.dump(doc_order, f, ensure_ascii=False, indent=2)

        # Save vectorizer and matrix
        with open(Path(save_dir) / 'tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(Path(save_dir) / 'tfidf_matrix.pkl', 'wb') as f:
            pickle.dump(self.tfidf_matrix, f)

        print(f"Saved index files to {save_dir}")

    def search(self, query: str, topk: int = 10):
        if self.tfidf_matrix is None:
            raise RuntimeError('Indexer not built. Call build() or load artifacts.')
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.tfidf_matrix)[0]
        idx = np.argsort(sims)[::-1]
        results = []
        for i in idx[:topk]:
            if sims[i] <= 0:
                continue
            doc = self.docs[i]
            results.append({'doc_id': doc['id'], 'url': doc.get('url'), 'score': float(sims[i])})
        return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs', default='../crawling/output_docs.jsonl', help='Path to docs jsonl')
    parser.add_argument('--out', default='index_output', help='Directory to save index')
    args = parser.parse_args()

    idx = Indexer(args.docs)
    idx.build(args.out)


if __name__ == '__main__':
    main()
