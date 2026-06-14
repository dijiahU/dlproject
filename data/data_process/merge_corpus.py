import hashlib
import json
from datetime import datetime, timezone
from urllib.parse import urlparse

# 配置文件路径
OLD_DOCUMENTS_FILE = '../processed/merged_documents.jsonl' #'../sist/jsonl/documents.jsonl'
OLD_CHUNKED_FILE = '../processed/merged_chunks.jsonl' #'../sist/jsonl/chunks.jsonl'
NEW_RAW_FILE = '../data_scrapy_new/sist_corpus.jsonl'
MERGED_OUTPUT_FILE = '../processed/merged_chunks.jsonl'
MERGED_DOCUMENTS_FILE = '../processed/merged_documents.jsonl'

# 假设的分块参数 (你需要根据你之前清洗旧数据的参数来调整)
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120

def generate_doc_id(url):
    """
    使用 URL 的 MD5 哈希值作为稳定的 document_id。
    这样如果以后重复爬取同一个 URL，生成的 ID 是一样的，方便溯源和强去重。
    """
    return hashlib.md5(url.encode('utf-8')).hexdigest()

def load_jsonl(path):
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def build_document_record(doc, document_id, run_id, fetched_at):
    url = doc.get('url', '')
    content = doc.get('content', '')
    host = urlparse(url).netloc or None
    text_chars = len(content)
    sha256 = hashlib.sha256(content.encode('utf-8')).hexdigest() if content else None
    valid_from = fetched_at.split('T', 1)[0] if fetched_at else None

    return {
        "id": document_id,
        "run_id": run_id,
        "url": url,
        "canonical_url": doc.get('canonical_url') or url,
        "title": doc.get('title', ''),
        "host": host,
        "category": doc.get('category'),
        "language": doc.get('language'),
        "content_type": doc.get('content_type', 'text/html'),
        "status_code": doc.get('status_code', 200),
        "fetched_at": fetched_at,
        "source_published_at": doc.get('source_published_at'),
        "valid_from": valid_from,
        "valid_until": doc.get('valid_until'),
        "validity_note": doc.get('validity_note', 'merged_from_new_raw'),
        "raw_path": doc.get('raw_path'),
        "text_path": doc.get('text_path'),
        "sha256": sha256,
        "depth": doc.get('depth', 0),
        "parent_url": doc.get('parent_url'),
        "text_chars": text_chars,
    }

def simple_chunk_text(text, chunk_size, chunk_overlap):
    """
    简单的滑动窗口分块函数。
    建议替换为你清洗旧数据时所使用的标准分块器 (例如 LangChain 的 RecursiveCharacterTextSplitter)。
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - chunk_overlap)
    return chunks

def merge_and_chunk_corpora():
    existing_urls = set()
    merged_data = []
    merged_documents = []
    max_doc_id = 0
    max_run_id = 0
    
    # ==========================================
    # 读取旧 documents，作为新 documents 的基础
    # ==========================================
    print("正在加载旧 documents 并建立文档索引...")
    try:
        old_documents = load_jsonl(OLD_DOCUMENTS_FILE)
        merged_documents.extend(old_documents)
        for item in old_documents:
            url = item.get('url')
            canonical_url = item.get('canonical_url')
            if url:
                existing_urls.add(url)
            if canonical_url:
                existing_urls.add(canonical_url)
            max_doc_id = max(max_doc_id, safe_int(item.get('id'), 0))
            max_run_id = max(max_run_id, safe_int(item.get('run_id'), 0))
    except FileNotFoundError:
        print(f"警告：未找到旧文件 {OLD_DOCUMENTS_FILE}，将只处理新文件。")
    
    # ==========================================
    # 第一步：读取旧文件，提取已有的 URL 和数据
    # ==========================================
    print("正在加载旧知识库并建立去重索引...")
    try:
        old_chunks = load_jsonl(OLD_CHUNKED_FILE)
        for item in old_chunks:
            merged_data.append(item) # 把旧数据先放进合并列表
            
            # 提取旧数据中的 URL。假设你的旧数据里保留了 'url' 或 'source' 字段
            url = item.get('url') # or item.get('source')
            if url:
                existing_urls.add(url)
            max_doc_id = max(max_doc_id, safe_int(item.get('document_id'), 0))
    except FileNotFoundError:
        print(f"警告：未找到旧文件 {OLD_CHUNKED_FILE}，将只处理新文件。")

    print(f"旧知识库加载完毕，当前包含 {len(existing_urls)} 个独立网页的块。")

    # ==========================================
    # 第二步：读取新文件，去重并分块打标签
    # ==========================================
    print("正在处理新爬取的数据...")
    new_doc_count = 0
    new_chunk_count = 0
    new_run_id = max_run_id + 1 if max_run_id else 1
    fetched_at = datetime.now(timezone.utc).isoformat(timespec='seconds')
    
    with open(NEW_RAW_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            url = doc.get('url')
            
            # 1. 文档级去重：如果这个网页已经在旧库里了，直接跳过
            if url in existing_urls:
                continue
            
            content = doc.get('content', '')
            if not content:
                continue

            existing_urls.add(url) # 避免新文件内部存在重复 URL
            # 2. 生成文档级 ID
            # 如果你之前的 document_id 是自增数字，你需要在这里改成取最大数字 + 1
            # 但工程上更推荐使用哈希或 UUID
            document_id = max_doc_id + 1  #generate_doc_id(url)
            max_doc_id += 1
            new_doc_count += 1

            document_item = build_document_record(doc, document_id, new_run_id, fetched_at)
            merged_documents.append(document_item)
            
            # 3. 对长文本进行分块 (Chunking)
            chunks = simple_chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
            
            # 4. 为每个 Chunk 打上规范的元数据标签 (Metadata)
            for index, chunk_text in enumerate(chunks):
                chunk_item = {
                    "document_id": document_id,
                    "chunk_id": f"{document_id}-{index}",
                    "chunk_index": index,
                    "url": url,
                    "title": doc.get('title', ''),
                    "text": chunk_text  # RAG 系统实际用于向量化的文本
                }
                merged_data.append(chunk_item)
                new_chunk_count += 1

    # ==========================================
    # 第三步：将所有数据写入合并后的新文件
    # ==========================================
    print(f"处理完成。新增了 {new_doc_count} 个独立网页，拆分为 {new_chunk_count} 个新数据块。")
    
    print(f"正在保存至 {MERGED_OUTPUT_FILE}...")
    write_jsonl(MERGED_OUTPUT_FILE, merged_data)
    
    print(f"正在保存至 {MERGED_DOCUMENTS_FILE}...")
    write_jsonl(MERGED_DOCUMENTS_FILE, merged_documents)

    print("合并成功！")

if __name__ == "__main__":
    merge_and_chunk_corpora()
