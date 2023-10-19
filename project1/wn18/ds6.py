import pathlib
import os
import pprint
import pandas as pd
import igraph as ig
import numpy as np
from pykeen.datasets import get_dataset
from pykeen.hpo import hpo_pipeline

dataset = get_dataset(dataset="WN18")
path = dataset.training.metadata['path']
table_m = pd.read_csv(path, sep='\t')
table_m.columns = ["out", "rel", "inn"]

num_records = len(table_m)
raw_graph = []
for i in range(num_records):
    raw_graph.extend([str(table_m.loc[i, "out"]), str(table_m.loc[i, "inn"])])
KG = ig.Graph(directed=True)
KG.add_vertices(list(set(raw_graph)))
KG.add_edges([(str(table_m.loc[i, "out"]), str(table_m.loc[i, "inn"])) for i in range(num_records)])
KG_M = np.array(KG.get_adjacency().data)

KG_pagerank = KG.pagerank(directed=True)
KG_pagerank_dict = {k: v for k,v in enumerate(KG_pagerank)}
KG_pagerank_sorted = sorted(KG_pagerank_dict.items(), key = lambda item: item[1], reverse = True)
KG_original = KG_pagerank_sorted

thresholds = [1, 10, 50, 100, 300, 500, 700, 800, 900, 1000, 2000, 3000, 5000]

for i in thresholds:

  KG_original_threshold = KG_original[:i]
  remove_entity = []

  for n in range(i):    
      v = KG.vs[KG_original_threshold[n][0]]["name"]
      remove_entity.append(v)

  # Form a new dataset

  path = '/home/kang/.data/pykeen/datasets/wn18/wordnet-mlj12/ds6/wordnet-mlj12-train.txt'
  table_m = pd.read_csv(path, sep='\t', header=None)
  table_m.columns = ["out", "rel", "inn"]

  for v in remove_entity:
      table_m = table_m[table_m.out != v]
      table_m = table_m[table_m.inn != v]

  table_m.to_csv(path, sep='\t', index=False, header=False)

  training = '/home/kang/.data/pykeen/datasets/wn18/wordnet-mlj12/ds6/wordnet-mlj12-train.txt'
  testing = '/home/kang/.data/pykeen/datasets/wn18/wordnet-mlj12/ds6/wordnet-mlj12-test.txt'
  validation = '/home/kang/.data/pykeen/datasets/wn18/wordnet-mlj12/ds6/wordnet-mlj12-valid.txt' 

  hpo_result = hpo_pipeline(
      n_trials = 1,  
      training = training,
      testing = testing,
      validation = validation,
      evaluator = "rankbased",
      loss = "crossentropy",
      negative_sampler_kwargs = {
        "num_negs_per_pos": 98
      },
      optimizer = "adam",
      optimizer_kwargs = {
        "lr": 0.05869073902981315
      },
      model = 'RotatE',
      model_kwargs = {
        "embedding_dim": 64
      },
      epochs = 100,
      training_kwargs = {
        "batch_size": 4096,
      },
  )

  hpo_result.save_to_directory('wn18_ratate_hpo_ds6_num'+str(i))