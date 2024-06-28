import pandas as pd
import re
import jieba
import jieba.posseg as pseg
import nltk
import matplotlib.pyplot as plt
import platform

from matplotlib.font_manager import FontProperties
from wordcloud import WordCloud
from nltk.corpus import stopwords
from snownlp import SnowNLP
from collections import Counter
from gensim import corpora, models

SG_WORDS_RAW_FILEPATH = 'src/sg_words.txt'
SG_WORDS_CLEAN_FILEPATH = 'src/sg_words_clean.txt'

FONT_PATH = 'src/Microsoft_YaHei_Bold.ttf'
# 读取文件并去重
with open(SG_WORDS_RAW_FILEPATH, 'r', encoding='utf-8') as file:
    unique_lines = set(file.readlines())

# 写回到新文件
with open(SG_WORDS_CLEAN_FILEPATH, 'w', encoding='utf-8') as file:
    file.writelines(sorted(unique_lines))

jieba.load_userdict(SG_WORDS_CLEAN_FILEPATH)


# 添加自定义词语的词性
jieba.add_word('好吃', tag='a')  # 假设你想将“好吃”标记为形容词

# get unique SG words
unique_words = []
for item in unique_lines:
    unique_word = item.replace('\n','')
    unique_words.append(unique_word)
unique_words = list(filter(None, unique_words))


def clean_text(text):
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)

    # 去除网址链接
    text = re.sub(r'http\S+', '', text)

    # 去除表情符号（根据实际情况可能需要定制正则表达式）
    text = re.sub(r'\[[^\[\]]+R\]', '', text)

    # 去掉无信息表述
    text = re.sub(r'[如|见]*图[片]*\w+', ' ', text)

    # 辨认和分割句子
    text = re.sub(r'(?<=[^\W\da-zA-Z])\s+(?=[^\W\da-zA-Z])', '。', text)
    text = re.sub(r'[！!？?.]', '。', text)

    # 去除特殊符号和标点（保留句号）
    text = re.sub(r'[^\w\s。]', '', text)

    # 去除空格和换行符
    text = re.sub(r'\s+', ' ', text).strip()

    # 去除无意义的数字和单字
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\b\w\b', '', text)

    # text = re.sub(r'[^\w\s,，]', '', text)

    return text


def remove_null_records(df, colname):
    df = df.dropna(subset=[colname])
    df = df[df[colname] != 'nan']
    df = df[df[colname] != 'NaN']
    df = df.reset_index(drop=True)
    return df


def get_cn_stopwords():
    # from github https://github.com/stopwords-iso/stopwords-zh
    stopwords_manual = [line.strip() for line in open('src/stopwords-zh.txt', 'r', encoding='UTF-8').readlines()]

    # from nltk
    nltk.download('stopwords')
    stopwords_cn_list = stopwords.words('chinese')

    stop_words = list(set(stopwords_manual) | set(stopwords_cn_list))
    return stop_words


def break_comments_into_words(df, colname_cmt):
    jieba.initialize()  # 确保jieba的字典已加载
    stopwords = get_cn_stopwords()

    # 创建新列
    df['all_words'] = ''
    df['verbs'] = ''
    df['nouns'] = ''
    df['adjectives'] = ''

    # 定义分词和词性标注函数
    def get_words(sentence):
        words = pseg.cut(sentence)
        all_words = []
        verbs = []
        nouns = []
        adjectives = []
        for word, flag in words:
            if word != '。' and word not in stopwords:
                all_words.append(word)
                if flag.startswith('v'):
                    verbs.append(word)
                elif flag.startswith('n') or word in unique_words:
                    nouns.append(word)
                elif flag.startswith('a'):
                    adjectives.append(word)
        return ' '.join(all_words), ' '.join(verbs), ' '.join(nouns), ' '.join(adjectives)

    # 应用函数，并将结果分配到新列
    df[['all_words', 'verbs', 'nouns', 'adjectives']] = df[colname_cmt].apply(lambda x: pd.Series(get_words(x)))

    return df


# 获取最常见的词汇
def get_most_common_words(df, words_colname, length=20, display=False):
    words = ' '.join(df[words_colname]).split()
    word_counts = Counter(words)
    common_words = word_counts.most_common(length)

    if display:
        print(f'top {length} common words ({words_colname}):\n', common_words, '\n-------\n')

        # 分别提取词语和它们的频率
        word_items = [item[0] for item in common_words]
        frequencies = [item[1] for item in common_words]

        # 创建柱状图
        plt.figure(figsize=(10, 8))  # 可以调整图的大小
        plt.bar(word_items, frequencies, color='skyblue')  # 可以调整颜色

        # 设置字体，使其在条形图上显示中文
        cn_font_prop = FontProperties(fname=FONT_PATH)
        plt.xticks(rotation=45, fontproperties=cn_font_prop)
        plt.yticks(fontproperties=cn_font_prop)

        # 添加标题和标签
        plt.title(f'Top {length} Most Common Words ({words_colname})')
        plt.xlabel('Words')
        plt.ylabel('Frequency')

        # 显示数值标签
        for i in range(len(frequencies)):
            plt.text(i, frequencies[i] + 10, str(frequencies[i]), ha='center')

        # 旋转x轴上的标签，以便它们更容易阅读
        plt.xticks(rotation=45)

        # 显示图表
        plt.tight_layout()  # 调整布局以适应标签
        plt.show()

    return common_words


# 中文情感分析分数
def append_sentiment_score(df, colnames):
    def get_sentiment_cn(text):
        try:
            s = SnowNLP(text)
            return s.sentiments
        except ZeroDivisionError:
            return None  # 或者选择一个合适的默认值，比如中性情感0.5

    for colname in colnames:
        df[f'sentiment_{colname}'] = df[colname].apply(get_sentiment_cn)

    return df


# 生成词云
def show_word_cloud(df, words_colname='all_words'):
    text = ' '.join(df[words_colname])
    wordcloud = WordCloud(font_path=FONT_PATH).generate(text)

    print(f'Word Cloud ({words_colname}):\n')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

df = pd.read_csv('src/Ctrip_clean.csv', encoding='utf_8_sig', index_col=False)
df['content'].drop_duplicates()
df['content'].dropna()


# 应用清洗函数
df['content'] = df['content'].apply(str)
df['cleaned_content'] = df['content'].apply(clean_text)
df = remove_null_records(df, 'cleaned_content')

# 查看清洗后的结果df
print(df['cleaned_content'])


#分句子
all_stc_list = []

for index, row in df.iterrows():
    sentences = row['cleaned_content'].split('。')
    sentences = list(filter(None, sentences))
    for s in sentences:
        all_stc_list.append({'review_sentence':s})

df_stc = pd.DataFrame(all_stc_list)
print(df_stc)

# 分词
df_cmt = break_comments_into_words(df, 'cleaned_content')
df_stc = break_comments_into_words(df_stc, 'review_sentence')
print(df_stc)
print(df_cmt)


# show common words
common_words_all = get_most_common_words(df_cmt, 'all_words', 15, 1)
common_words_all = get_most_common_words(df_cmt, 'verbs', 15, 1)
common_words_all = get_most_common_words(df_cmt, 'nouns', 15, 1)
common_words_all = get_most_common_words(df_cmt, 'adjectives', 15, 1)

# 建立词典
dictionary = corpora.Dictionary(df_cmt['all_words'].apply(lambda x: x.split()))

# 建立语料库
corpus = [dictionary.doc2bow(text.split()) for text in df_cmt['all_words']]

# LDA模型训练
lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)

# 打印主题
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

df_cmt.to_csv("ctrip/Ctrip_clean_review.csv", mode='a+', index=False, encoding='utf_8_sig')
