from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")

sentences = [
      "《深度学习》这门课的任课老师是谁？",   # 问句
      "深度学习课程的授课教师信息",            # 和上一句意思接近
      "上海科技大学一共有几个学院？",          # 完全不同的话题
      "今天天气很好适合出去散步",              # 和学校毫无关系
  ]
embeddings = model.encode(sentences,normalize_embeddings=True)

print("向量矩阵的形状：",embeddings.shape)

similarity = model.similarity(embeddings,embeddings)
print("相似度矩阵:")
print(similarity)

# 6) 重点：用第 0 句去和其余每句比
print()   
print("以第 0 句『谁是任课老师』为基准，它和各句的相似度：")
for i in range(len(sentences)):
    print(f"  与第 {i} 句的相似度: {similarity[0][i]:.3f}   ->  {sentences[i]}")
