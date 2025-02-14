from flask import Flask, request, render_template, send_file, redirect, url_for
import os
import pandas as pd
import uuid
import tcvectordb
from tcvdb_text.encoder import BM25Encoder
from tcvectordb.model.document import Document
from tcvectordb.model.param.search import HybridSearchParam, AnnSearch, KeywordSearch, WeightedRerank

app = Flask(__name__)

# 配置上传文件的保存路径和结果导出路径
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 初始化向量数据库客户端
vdb_url = 'http://lb-o1jzbc6h-rmkf8txyz0c17dq5.clb.ap-shanghai.tencentclb.com:10000'      # 替换为你的连接URL
vdb_key = 'R4fThwO4VJhgp9m70YvKDprZ8IrIuzBi1b43Gb9i'      # 替换为你的连接KEY

client = tcvectordb.RPCVectorDBClient(
    url=vdb_url,
    key=vdb_key,
    username='root',
    read_consistency=tcvectordb.ReadConsistency.EVENTUAL_CONSISTENCY,
    timeout=30
)

db_name = 'web_demo_db'
collection_name = 'web_demo_collection'

# 初始化BM25 Encoder
encoder = BM25Encoder.default('zh')

# 在首次请求前设置数据库和集合
@app.before_first_request
def setup_vector_db():
    try:
        client.drop_database(db_name)
    except Exception:
        pass  # 如果数据库不存在则跳过

    db = client.create_database(db_name)

    # 定义集合的索引结构
    index = tcvectordb.Index()
    index.add(tcvectordb.FilterIndex('id', tcvectordb.FieldType.String, tcvectordb.IndexType.PRIMARY_KEY))
    index.add(tcvectordb.VectorIndex(
        name='vector',
        dimension=768,
        index_type=tcvectordb.IndexType.HNSW,
        metric_type=tcvectordb.MetricType.IP,
        params=tcvectordb.HNSWParams(m=16, efconstruction=200)
    ))

    # 创建 Collection
    db.create_collection(
        name=collection_name,
        shard=1,
        replicas=1,
        description='Web Demo Collection',
        index=index
    )

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 处理上传的查询文件和内容片段文件
        queries_file = request.files.get('queries')
        contents_file = request.files.get('contents')

        if not queries_file or not contents_file:
            return "请上传查询文件和内容片段文件。", 400

        # 保存上传的文件
        queries_path = os.path.join(app.config['UPLOAD_FOLDER'], queries_file.filename)
        contents_path = os.path.join(app.config['UPLOAD_FOLDER'], contents_file.filename)
        queries_file.save(queries_path)
        contents_file.save(contents_path)

        # 读取CSV文件
        queries_df = pd.read_csv(queries_path)
        contents_df = pd.read_csv(contents_path)

        # 向量化查询和内容片段
        queries_vectors = encoder.encode_texts(queries_df['query'].tolist())
        contents_vectors = encoder.encode_texts(contents_df['content'].tolist())

        # 准备要插入向量数据库的文档
        documents = []
        for idx, row in contents_df.iterrows():
            doc = {
                "id": str(uuid.uuid4()),
                "vector": contents_vectors[idx],
                "sparse_vector": contents_vectors[idx],
                "text": row['content']
            }
            documents.append(doc)

        # 批量插入内容片段到向量数据库
        client.upsert(
            database_name=db_name,
            collection_name=collection_name,
            documents=documents
        )

        # 等待数据索引完成
        import time
        time.sleep(5)  # 根据实际情况调整等待时间

        # 进行混合检索
        results = []
        for idx, row in queries_df.iterrows():
            query_vector = queries_vectors[idx]
            query_sparse_vector = encoder.encode_queries([row['query']])[0]

            hybrid_search_param = HybridSearchParam(
                ann=[AnnSearch(field_name="vector", data=query_vector)],
                match=[KeywordSearch(field_name="sparse_vector", data=query_sparse_vector)],
                rerank=WeightedRerank(field_list=['vector', 'sparse_vector'], weight=[0.9, 0.1]),
                retrieve_vector=False,
                limit=10
            )

            search_results = client.hybrid_search(
                database_name=db_name,
                collection_name=collection_name,
                param=hybrid_search_param
            )

            for doc in search_results:
                results.append({
                    'query': row['query'],
                    'content': doc['text'],
                    'score': doc['score']
                })

        # 将结果保存为CSV
        results_df = pd.DataFrame(results)
        results_file = os.path.join(RESULTS_FOLDER, f'results_{uuid.uuid4()}.csv')
        results_df.to_csv(results_file, index=False)

        return redirect(url_for('download_file', filename=os.path.basename(results_file)))

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(RESULTS_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)