import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------
# 缓存模型和计算结果，提高重复运行效率
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_dense_model():
    # 加载 SentenceTransformer 模型（这里使用 all-MiniLM-L6-v2）
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

@st.cache_resource(show_spinner=False)
def compute_dense_embeddings(model, texts):
    # 计算稠密向量
    return model.encode(texts, convert_to_numpy=True)

@st.cache_resource(show_spinner=False)
def compute_tfidf_matrix(texts):
    # 使用 TfidfVectorizer 计算文档的 TF-IDF 矩阵
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix

# -------------------------------
# Streamlit 主程序
# -------------------------------
st.title("混合检索 Demo")
st.write("本 demo 演示基于稠密向量（语义检索）和稀疏向量（关键词检索）的混合检索。请导入或输入一批 query 及内容片段，系统将计算各自的匹配分，并支持导出结果表格。")

st.markdown("---")
# 输入方式选择
input_mode = st.radio("请选择输入方式：", ("手动输入", "上传 CSV 文件"))

if input_mode == "手动输入":
    st.subheader("手动输入")
    queries_text = st.text_area("请输入查询（每行一个）：", height=150)
    docs_text = st.text_area("请输入内容片段（每行一个）：", height=150)
    queries = [line.strip() for line in queries_text.splitlines() if line.strip()]
    docs = [line.strip() for line in docs_text.splitlines() if line.strip()]
else:
    st.subheader("上传 CSV 文件")
    col1, col2 = st.columns(2)
    with col1:
        query_file = st.file_uploader("上传查询 CSV 文件（至少包含一列名为 'query'）", type=["csv"])
    with col2:
        doc_file = st.file_uploader("上传内容 CSV 文件（至少包含一列名为 'segment'）", type=["csv"])
    queries = []
    docs = []
    if query_file is not None:
        try:
            queries_df = pd.read_csv(query_file)
            if 'query' not in queries_df.columns:
                st.error("查询文件中必须包含 'query' 列")
            else:
                queries = queries_df['query'].dropna().astype(str).tolist()
        except Exception as e:
            st.error(f"读取查询文件出错：{e}")
    if doc_file is not None:
        try:
            docs_df = pd.read_csv(doc_file)
            if 'segment' not in docs_df.columns:
                st.error("内容文件中必须包含 'segment' 列")
            else:
                docs = docs_df['segment'].dropna().astype(str).tolist()
        except Exception as e:
            st.error(f"读取内容文件出错：{e}")

st.markdown("---")
# 权重设置
st.subheader("设置检索权重")
dense_weight = st.slider("稠密向量权重（语义检索）", 0.0, 1.0, 0.9, 0.05)
sparse_weight = st.slider("稀疏向量权重（关键词检索）", 0.0, 1.0, 0.1, 0.05)

# 开始处理
if st.button("运行混合检索"):

    if not queries:
        st.error("请提供至少一个查询！")
    elif not docs:
        st.error("请提供至少一个内容片段！")
    else:
        st.info("开始向量化并计算匹配分，请稍候...")
        # 加载并计算稠密嵌入
        dense_model = load_dense_model()
        with st.spinner("计算稠密向量..."):
            dense_docs = compute_dense_embeddings(dense_model, docs)
            dense_queries = compute_dense_embeddings(dense_model, queries)
        dense_similarities = cosine_similarity(dense_queries, dense_docs)
        
        # 稀疏向量处理：利用 TF-IDF
        with st.spinner("计算稀疏向量 (TF-IDF)..."):
            tfidf_vectorizer, tfidf_docs = compute_tfidf_matrix(docs)
            tfidf_queries = tfidf_vectorizer.transform(queries)
            sparse_similarities = cosine_similarity(tfidf_queries, tfidf_docs)
        
        # 混合得分
        hybrid_similarities = dense_weight * dense_similarities + sparse_weight * sparse_similarities
        
        # 构造结果表格，每一行为一个 (query, segment) 对应的各项得分
        result_rows = []
        for i, query in enumerate(queries):
            for j, doc in enumerate(docs):
                result_rows.append({
                    "query": query,
                    "segment": doc,
                    "dense_score": dense_similarities[i][j],
                    "sparse_score": sparse_similarities[i][j],
                    "hybrid_score": hybrid_similarities[i][j]
                })
        result_df = pd.DataFrame(result_rows)
        
        st.success("混合检索完成！")
        st.subheader("检索结果")
        st.dataframe(result_df)
        
        # 下载结果表格
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="导出结果为 CSV",
            data=csv,
            file_name='hybrid_search_results.csv',
            mime='text/csv'
        ) 
