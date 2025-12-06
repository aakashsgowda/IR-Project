# IR Project Report

**Abstract:**
- Development summary: Implemented a Scrapy-based crawler that stores crawled pages as JSON Lines (`output_docs.jsonl`), a Scikit-Learn based indexer that builds TF-IDF representations and an inverted index, and a Flask-based query processor that accepts CSV queries and returns Top-K ranked results.
- Objectives: provide a minimal IR pipeline (crawl → index → query) with TF-IDF + cosine ranking and clear extension points for embeddings/FAISS and query expansion.
- Next steps: add distributed crawling via `scrapyd`, integrate word2vec/transformer embeddings, and add spelling correction / query expansion.

**Overview:**
- Solution outline: `crawling/` contains Scrapy spiders. `Scikit/` contains the indexer which outputs `index_output/` artifacts. `zFlask/` contains the query processor server.
- Relevant literature: classic TF-IDF and vector-space model (Salton & McGill), cosine similarity ranking, modern semantic search using embeddings (Mikolov et al.; Reimers & Gurevych for SBERT), FAISS (Johnson et al.).

**Design:**
- System capabilities: seed-based crawl, configurable max pages/depth, TF-IDF indexing, cosine-ranked retrieval, CSV-batch query processing.
- Interactions: Scrapy writes `output_docs.jsonl` → Scikit indexer reads it and writes `index_output/` → Flask app loads index and answers queries.

**Architecture:**
- Software components: `GenericCrawler` (Scrapy spider), `indexer.py` (builds index), `zFlask/app.py` (query API).
- Interfaces: crawler output: JSON Lines with `id,url,depth,text`; index artifacts: `inverted_index.json`, `doc_meta.json`, `doc_order.json`, pickled `tfidf_vectorizer.pkl` and `tfidf_matrix.pkl`.

**Operation:**
- Installation: create a Python 3.12+ virtualenv and install `pip install -r requirements.txt`.
- Crawl command (example):
```
cd Project/crawling
scrapy crawl generic_crawler -a start_url=https://books.toscrape.com -a max_pages=50 -a max_depth=3
```
- Build index:
```
cd Project/Scikit
python3 indexer.py --docs ../crawling/output_docs.jsonl --out index_output
```
- Run Flask server:
```
cd Project/zFlask
python3 app.py
```
- Query via CSV upload (field `file`) to `/search_csv`.

**Conclusion:**
- Successes: Core pipeline implemented; TF-IDF + cosine ranking works end-to-end. Artifacts are saved for reuse by the Flask server.
- Caveats: The crawler is polite (respects robots.txt) and is single-machine. Indexer uses pickles for matrix storage; transfer or versioning may require care. Flask app assumes index artifacts exist.

**Data Sources:**
- Example seed used in repository: `https://books.toscrape.com/` (public scraping sandbox). Replace with your own domain respecting robots.txt and terms.

**Test Cases:**
- Sanity: crawl 5 pages, build index, run a query CSV with few queries and verify returned ranked docs.
- Suggested harness: small shell scripts that run the three commands above and check for created files.

**Source Code:**
- Located in the repository; key files: `crawling/crawling/spiders/GenericCrawler.py`, `Scikit/indexer.py`, `zFlask/app.py`.
- Dependencies: `requirements.txt`.

**Bibliography (selected):**
- Salton, G., & McGill, M. J. (1983). Introduction to Modern Information Retrieval. McGraw-Hill.
- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). "Efficient Estimation of Word Representations in Vector Space." arXiv preprint arXiv:1301.3781.
- Johnson, J., Douze, M., & Jégou, H. (2019). "Billion-scale similarity search with GPUs." IEEE Transactions on Big Data.
- Reimers, N., & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." EMNLP.
