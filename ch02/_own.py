# なんかどっか間違えてて正常な値になってないけど放置
import sys

sys.path.append("..")
import numpy as np
from common.util import preprocess

text = "you say goobye and i say hello."
corpus, word_to_id, id_to_word = preprocess(text)


def create_co_matrix(corpus, vocab_size, window_size=1):
    """共起行列の作成

    :param corpus: コーパス（単語IDのリスト）
    :param vocab_size:語彙数
    :param window_size:ウィンドウサイズ（ウィンドウサイズが1のときは、単語の左右1単語がコンテキスト）
    :return: 共起行列
    """
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


# print(create_to_matrix(corpus, len(corpus)))


def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x**2)) + eps)
    ny = y / (np.sqrt(np.sum(x**2)) + eps)
    return np.dot(nx, ny)


vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

c0 = C[word_to_id["you"]]  # 「you」の単語ベクトル
c1 = C[word_to_id["i"]]  # 「i」の単語ベクトル
print(cos_similarity(c0, c1))
