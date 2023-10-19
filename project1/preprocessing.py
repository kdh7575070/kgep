import pandas as pd
import pykeen
from pykeen.datasets import get_dataset
import igraph as ig

# Load CSV and rename columns
# path = 'C:\\Users\\USER\\Desktop\\pp\\WN18RR.csv'
# table_m = pd.read_csv(path)

# Load CSV and rename columns using pykeen

DATA = "FB15k"
dataset = get_dataset(dataset = DATA)
path = dataset.training.metadata['path']

table_m = pd.read_csv(path, sep ='\t', header = None)
table_m.columns = ["out", "rel", "inn"]
# f = open("C:\\Users\\USER\\Desktop\\pp\\FB15K_result.txt", mode = 'w', encoding = 'UTF-8')


##### degree centrality #####

# degree centrality - relationship degree (frequency of each relationship)
table_cnt_rel = table_m.groupby("rel").size().reset_index(name="n").sort_values(by="n", ascending=False)
print("degree centrality REL 10:\n", table_cnt_rel.head(10))
print("degree centrality REL M:", table_cnt_rel["n"].mean())
print("degree centrality REL 10M:", table_cnt_rel.head(10)["n"].mean())

# # degree centrality - in-degree (avg in-coming arrow from each entity)
table_cnt_in = table_m.groupby("inn").size().reset_index(name="n").sort_values(by="n", ascending=False)
print("\ndegree centrality IN 10:\n", table_cnt_in.head(10))
print("degree centrality IN M:", table_cnt_in["n"].mean())
print("degree centrality IN 10M:", table_cnt_in.head(10)["n"].mean())

# # degree centrality - out-degree (avg out-going arrow from each entity)
table_cnt_out = table_m.groupby("out").size().reset_index(name="n").sort_values(by="n", ascending=False)
print("\ndegree centrality OUT 10:\n", table_cnt_out.head(10))
print("degree centrality OUT M:", table_cnt_out["n"].mean())
print("degree centrality OUT 10M:", table_cnt_out.head(10)["n"].mean())

# # degree centrality - overall degree (avg arrow from each entity)
table_cnt_all = pd.merge(table_cnt_out, table_cnt_in, left_on="out", right_on="inn", how="inner")
table_cnt_all["n"] = table_cnt_all["n_x"] + table_cnt_all["n_y"]
table_cnt_all = table_cnt_all[["out", "n"]].rename(columns={"out": "entity"}).sort_values(by="n", ascending=False)
print("\ndegree centrality INOUT 10:\n", table_cnt_all.head(10))
print("degree centrality INOUT M:", table_cnt_all["n"].mean())
print("degree centrality INOUT 10M:", table_cnt_all.head(10)["n"].mean())

# # Graph and adjacency matrix building
num_records = len(table_m)
raw_graph = []
for i in range(num_records):
    raw_graph.extend([str(table_m.loc[i, "out"]), str(table_m.loc[i, "inn"])])
KG = ig.Graph(directed=True)
KG.add_vertices(list(set(raw_graph)))
KG.add_edges([(str(table_m.loc[i, "out"]), str(table_m.loc[i, "inn"])) for i in range(num_records)])
# KG_M = np.array(KG.get_adjacency().data)
# ig.plot(KG)


# ##### closeness centrality #####
KG_closeness = KG.closeness()
KG_closeness_dict = {k: v for k,v in enumerate(KG_closeness)}
KG_closeness_sorted = sorted(KG_closeness_dict.items(), key = lambda item: item[1], reverse = True)
KG_closeness_sorted_10 = KG_closeness_sorted[:10]
KG_closeness_sorted_10_v = []

for i in range(10):    
    v = KG.vs[KG_closeness_sorted_10[i][0]]["name"]
    KG_closeness_sorted_10_v.append((v, KG_closeness_sorted_10[i][1]))
print("\ncloseness centrality 10 :", KG_closeness_sorted_10_v)

# ##### eigen centrality #####
KG_eigen = KG.eigenvector_centrality(directed=True)
KG_eigen_dict = {k: v for k,v in enumerate(KG_eigen)}
KG_eigen_sorted = sorted(KG_eigen_dict.items(), key = lambda item: item[1], reverse = True)
KG_eigen_sorted_10 = KG_eigen_sorted[:10]
KG_eigen_sorted_10_v = []

for i in range(10):    
    v = KG.vs[KG_eigen_sorted_10[i][0]]["name"]
    KG_eigen_sorted_10_v.append((v, KG_eigen_sorted_10[i][1]))
print("\neigen centrality 10 :", KG_eigen_sorted_10_v)


# ##### betweenness centrality #####
KG_betweenness = KG.betweenness(directed=True)
KG_betweenness_dict = {k: v for k,v in enumerate(KG_betweenness)}
KG_betweenness_sorted = sorted(KG_betweenness_dict.items(), key = lambda item: item[1], reverse = True)
KG_betweenness_sorted_10 = KG_betweenness_sorted[:10]
KG_betweenness_sorted_10_v = []

for i in range(10):    
    v = KG.vs[KG_betweenness_sorted_10[i][0]]["name"]
    KG_betweenness_sorted_10_v.append((v, KG_betweenness_sorted_10[i][1]))
print("\nbetweenness centrality 10 :", KG_betweenness_sorted_10_v)


# ##### pagerank #####
KG_pagerank = KG.pagerank(directed=True)
KG_pagerank_dict = {k: v for k,v in enumerate(KG_pagerank)}
KG_pagerank_sorted = sorted(KG_pagerank_dict.items(), key = lambda item: item[1], reverse = True)
KG_pagerank_sorted_10 = KG_pagerank_sorted[:10]
KG_pagerank_sorted_10_v = []

for i in range(10):    
    v = KG.vs[KG_pagerank_sorted_10[i][0]]["name"]
    KG_pagerank_sorted_10_v.append((v, KG_pagerank_sorted_10[i][1]))
print("\npagerank centrality 10 :", KG_pagerank_sorted_10_v)


# ##### harmonic centrality #####
KG_harmonic = KG.harmonic_centrality()
KG_harmonic_dict = {k: v for k,v in enumerate(KG_harmonic)}
KG_harmonic_sorted = sorted(KG_harmonic_dict.items(), key = lambda item: item[1], reverse = True)
KG_harmonic_sorted_10 = KG_harmonic_sorted[:10]
KG_harmonic_sorted_10_v = []

for i in range(10):    
    v = KG.vs[KG_harmonic_sorted_10[i][0]]["name"]
    KG_harmonic_sorted_10_v.append((v, KG_harmonic_sorted_10[i][1]))
print("\nharmonic centrality 10 :", KG_harmonic_sorted_10_v)


print()
print()


##### self-rotation #####
self_rot_cnt = sum(1 for i in range(num_records) if table_m.loc[i, "out"] == table_m.loc[i, "inn"])
print(self_rot_cnt)

##### rotation #####
# rot_cnt = 0
# for i in range(num_records):
#     for j in range(i, num_records):
#         if table_m.loc[i, "out"] == table_m.loc[j, "inn"] and table_m.loc[i, "inn"] == table_m.loc[j, "out"]:
#             rot_cnt += 1
#             print(rot_cnt)
#             break
