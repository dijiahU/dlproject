import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME  = "BAAI/bge-m3"
CHUNKS_PATH = "rag/data/chunks.jsonl"
INDEX_PATH  = "rag/index/faiss.index"
META_PATH   = "rag/index/meta.jsonl"
MAX_CHUNKS  = None

def load_chunks(path,limit = None):
    chunks=[]
    with open(path,encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
            if limit is not None and len(chunks) >= limit:
                break 
    return chunks

def build_index(chunks,model):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size = 32,
        show_progress_bar = True
    )

    embeddings = np.asarray(embeddings,dtype = "float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_meta(chunks, path):
    """保存每段的元信息（标题/链接/正文等），将来检索到第 i 条时能查回原文。"""
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            meta = {
                "chunk_id": c["chunk_id"],
                "title":    c["title"],
                "url":      c["url"],
                "category": c["category"],
                "text":     c["text"],
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    chunks = load_chunks(CHUNKS_PATH, limit=MAX_CHUNKS)
    print(f"加载了 {len(chunks)} 段文本")

    model = SentenceTransformer(MODEL_NAME)
    index = build_index(chunks, model)
    print("索引里的向量数量:", index.ntotal)

    faiss.write_index(index, INDEX_PATH)         # 把索引存成文件
    save_meta(chunks, META_PATH)                 # 把元信息存成文件
    faiss.write_index(index, INDEX_PATH)         # 把索引存成文件
    save_meta(chunks, META_PATH)                 # 把元信息存成文件
    print("✓ 索引已保存:", INDEX_PATH)
    print("✓ 元信息已保存:", META_PATH)