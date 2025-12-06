import csv
import io
import json
from pathlib import Path
from flask import Flask, request, send_file, jsonify, render_template
import pickle
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__, template_folder='templates', static_folder='static')

# Location where the indexer saved its artifacts
DATA_DIR = Path(__file__).resolve().parents[1] / 'Scikit' / 'index_output'


_INDEX_CACHE = {}


def load_index():
    vec_path = DATA_DIR / 'tfidf_vectorizer.pkl'
    mat_path = DATA_DIR / 'tfidf_matrix.pkl'
    meta_path = DATA_DIR / 'doc_meta.json'
    order_path = DATA_DIR / 'doc_order.json'
    if not vec_path.exists() or not mat_path.exists() or not meta_path.exists() or not order_path.exists():
        raise FileNotFoundError('Index artifacts not found. Run the indexer first.')
    # Cache loaded artifacts to avoid repeated disk I/O
    if DATA_DIR in _INDEX_CACHE:
        return _INDEX_CACHE[DATA_DIR]

    with open(vec_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(mat_path, 'rb') as f:
        tfidf_matrix = pickle.load(f)
    with open(meta_path, 'r', encoding='utf-8') as f:
        doc_meta = json.load(f)
    with open(order_path, 'r', encoding='utf-8') as f:
        doc_order = json.load(f)

    _INDEX_CACHE[DATA_DIR] = (vectorizer, tfidf_matrix, doc_meta, doc_order)
    return vectorizer, tfidf_matrix, doc_meta, doc_order


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/search_csv', methods=['POST'])
def search_csv():
    # Expect a multipart upload with file field 'file' containing CSV with header 'qid,query'
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded (field name `file`)'}), 400
    file = request.files['file']
    stream = io.StringIO(file.stream.read().decode('utf-8'))
    reader = csv.DictReader(stream)
    rows = list(reader)
    if not rows:
        return jsonify({'error': 'CSV empty or invalid. Expect header with `qid,query`.'}), 400

    try:
        vectorizer, tfidf_matrix, doc_meta = load_index()
    except Exception as e:
        return jsonify({'error': f'Failed to load index: {e}'}), 500

    out_stream = io.StringIO()
    writer = csv.writer(out_stream)
    writer.writerow(['qid', 'rank', 'doc_id', 'url', 'score'])

    for row in rows:
        qid = row.get('qid') or row.get('id') or ''
        query = row.get('query')
        if not query:
            continue
        qv = vectorizer.transform([query])
        sims = cosine_similarity(qv, tfidf_matrix)[0]
        idxs = sims.argsort()[::-1]
        rank = 1
        for i in idxs[:10]:
            if sims[i] <= 0:
                continue
            # find doc id from doc_meta; doc_meta keys are doc_id
            # order of tfidf_matrix matches order used when building indexer
            # For simplicity, the indexer stored docs in the same order in tfidf_matrix.pkl
            # We will attempt to read doc ids from the inverted index or assume indexer saved same ordering
            # To keep things simple here we load doc_meta and map by index order using a sidefile 'doc_order.json' if present
            # Fallback: write doc_id as index
            doc_id = str(i)
            # attempt to load doc order
            doc_order_path = DATA_DIR / 'doc_order.json'
            if doc_order_path.exists():
                with open(doc_order_path, 'r', encoding='utf-8') as f:
                    doc_order = json.load(f)
                doc_id = doc_order[i]
            url = doc_meta.get(doc_id, {}).get('url', '')
            writer.writerow([qid, rank, doc_id, url, float(sims[i])])
            rank += 1

    out_stream.seek(0)
    return send_file(io.BytesIO(out_stream.getvalue().encode('utf-8')), mimetype='text/csv', as_attachment=True, download_name='results.csv')


# Simple web UI: form for single query + Top-K results
@app.route('/', methods=['GET'])
def home():
        return render_template('home.html')


@app.route('/search', methods=['GET'])
def search():
    q = request.args.get('q', '').strip()
    try:
        per_page = int(request.args.get('k', '10'))
    except ValueError:
        per_page = 10
    try:
        page = int(request.args.get('page', '1'))
    except ValueError:
        page = 1

    if not q:
        return render_template('home.html', error='Please enter a query')

    try:
        vectorizer, tfidf_matrix, doc_meta, doc_order = load_index()
    except Exception as e:
        return render_template('home.html', error=f'Failed to load index: {e}')

    qv = vectorizer.transform([q])
    sims = cosine_similarity(qv, tfidf_matrix)[0]
    idxs = sims.argsort()[::-1]

    results = []
    for i in idxs:
        score = float(sims[i])
        if score <= 0:
            continue
        try:
            doc_id = doc_order[i]
        except Exception:
            doc_id = str(i)
        url = doc_meta.get(doc_id, {}).get('url', '')
        results.append({'doc_id': doc_id, 'url': url, 'score': score})

    # Pagination
    total = len(results)
    start = (page - 1) * per_page
    end = start + per_page
    page_results = results[start:end]
    total_pages = (total + per_page - 1) // per_page if per_page > 0 else 1

    return render_template('results.html', query=q, results=page_results, page=page, per_page=per_page, total=total, total_pages=total_pages)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
