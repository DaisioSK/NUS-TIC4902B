import pandas as pd
import numpy as np
from apyori import apriori
import networkx as nx
import matplotlib.pyplot as plt
from googletrans import Translator
from deep_translator import GoogleTranslator
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.font_manager import FontProperties


import jieba
from collections import Counter
# Load your dataset
df_cmt = pd.read_csv('ctrip/Ctrip_clean_review.csv', encoding='utf_8_sig', index_col=False)


translator = Translator()
font_path = 'src/Microsoft_YaHei_Bold.ttf'  # 请将这里的路径替换为你选择的中文字体路径
font = FontProperties(fname=font_path, size=12)
# Text preprocessing for frequent itemset mining using Jieba for tokenization
def process_all_words(value):
    if isinstance(value, str):
        return value.split()
    return []


def find_association_rules(transaction_col, min_support=0.01, min_confidence=0.3, min_lift=3, min_length=2):
    transactions = transaction_col.apply(process_all_words)
    transactions = [x for x in transactions if x != []]
    # transactions = [translate_to_english(x) for x in transactions]
    association_rules = apriori(transactions, min_support=min_support, min_confidence=min_confidence, min_lift=min_lift,
                                min_length=min_length)
    association_results = list(association_rules)

    G = nx.DiGraph()
    rules_summary = [{
        'antecedent': list(item[2][0][0]),
        'consequent': list(item[2][0][1]),
        'support': round(item[1], 4),
        'confidence': round(item[2][0][2], 4),
        'lift': round(item[2][0][3], 4),
    } for item in association_results]

    print(f'Success. {len(rules_summary)} rules found.')
    print("=====================================")
    for rule in rules_summary:
        print(f"Rule: {rule['antecedent']} => {rule['consequent']}")
        print("Support: " + str(rule['support']))
        print("Confidence: " + str(rule['confidence']))
        print("Lift: " + str(rule['lift']))
        print("------------------------------------")
        for ant in rule['antecedent']:
            for con in rule['consequent']:
                G.add_edge(ant, con, weight=rule['lift'], support=rule['support'], confidence=rule['confidence'])

    # 创建自定义颜色映射，最浅的颜色为原来颜色的一半深度，而不是白色
    colors = plt.cm.Blues(np.linspace(0.5, 1, 256))
    new_cmap = LinearSegmentedColormap.from_list('new_blues', colors)

    # 画图
    pos = nx.spring_layout(G, k=1, iterations=15)  # 调整布局参数 k 和迭代次数，使图更紧凑
    plt.figure(figsize=(12, 12))  # 增大图形尺寸

    # 节点颜色和大小
    node_color = 'skyblue'
    node_size = 3000

    # 边的宽度与颜色
    edges = G.edges(data=True)
    # edge_width = [d['weight'] for (u, v, d) in edges]
    edge_color = [d['weight'] for (u, v, d) in edges]

    # 画节点和边
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color)

    # 重新计算边的终点位置，使箭头不会被节点遮挡
    def shorten_edge(pos, src, dst, shrink_factor=0.1):
        """缩短边，使箭头不会被节点遮挡"""
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        new_x2 = x1 + (x2 - x1) * (1 - shrink_factor)
        new_y2 = y1 + (y2 - y1) * (1 - shrink_factor)
        return (x1, y1), (new_x2, new_y2)

    new_edges = []
    for src, dst, data in edges:
        new_edges.append((*shorten_edge(pos, src, dst), data))

    # 画边，缩短后的位置
    for ((x1, y1), (x2, y2), data) in new_edges:
        plt.arrow(x1, y1, x2 - x1, y2 - y1,
                  color=new_cmap(Normalize(vmin=min(edge_color), vmax=max(edge_color))(data['weight'])),
                  alpha=0.8, head_width=0.05, head_length=0.1, length_includes_head=True, width=0.01)
        # 在边的中点显示 lift 值
        plt.text((x1 + x2) / 2, (y1 + y2) / 2, f"{data['weight']:.2f}", fontsize=10, fontproperties=font,
                 horizontalalignment='center', verticalalignment='center')

    # 使用 plt.text 绘制标签并指定字体
    for node, (x, y) in pos.items():
        plt.text(x, y, node, fontsize=12, fontproperties=font, horizontalalignment='center', verticalalignment='center')

    # 调整颜色映射，使最浅的颜色也能看清
    norm = Normalize(vmin=min(edge_color), vmax=max(edge_color))
    sm = plt.cm.ScalarMappable(cmap=new_cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca())

    plt.title('Association Rules Network', fontproperties=font)
    plt.show()


# find_association_rules(data['content'],0.005,0.8,20)
find_association_rules(df_cmt['content'],0.005,0.8,20)
find_association_rules(df_cmt['all_words'],0.008,0.8,20)
find_association_rules(df_cmt['nouns'],0.006,0.8,20)
find_association_rules(df_cmt['verbs'],0.005,0.8,20)
find_association_rules(df_cmt['adjectives'],0.001,0.4,10)

