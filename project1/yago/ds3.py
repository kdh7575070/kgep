import pathlib
import os
import pprint
import pandas as pd
import igraph as ig
import numpy as np
from pykeen.datasets import get_dataset
from pykeen.hpo import hpo_pipeline

dataset = get_dataset(dataset="yago310")
path = dataset.training.metadata['path']
table_m = pd.read_csv(path, sep='\t')
table_m.columns = ["out", "rel", "inn"]

table_cnt_in = table_m.groupby("inn").size().reset_index(name="n").sort_values(by="n", ascending=False)
table_cnt_out = table_m.groupby("out").size().reset_index(name="n").sort_values(by="n", ascending=False)
table_cnt_all = pd.merge(table_cnt_out, table_cnt_in, left_on="out", right_on="inn", how="inner")
table_cnt_all["n"] = table_cnt_all["n_x"] + table_cnt_all["n_y"]
table_cnt_all = table_cnt_all[["out", "n"]].rename(columns={"out": "entity"}).sort_values(by="n", ascending=False)

thresholds = [1, 10, 100, 300, 500, 700, 800, 900, 1000, 2000, 3000]

for i in thresholds:

  remove_entity = []
  for n in range(i):
      remove_entity.append(table_cnt_all.head(i).iat[n,0]) #tail for bottom

  # Form a new dataset

  path = '/home/kang/.data/pykeen/datasets/yago310/ds3/train.txt'
  table_m = pd.read_csv(path, sep='\t', header=None)
  table_m.columns = ["out", "rel", "inn"]

  for v in remove_entity:
      table_m = table_m[table_m.out != v]
      table_m = table_m[table_m.inn != v]

  table_m.to_csv(path, sep='\t', index=False, header=False)

  training = '/home/kang/.data/pykeen/datasets/yago310/ds3/train.txt'
  testing = '/home/kang/.data/pykeen/datasets/yago310/ds3/test.txt'
  validation = '/home/kang/.data/pykeen/datasets/yago310/ds3/valid.txt'


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
          "lr": 0.010475585856302566
        },
        model = 'RotatE',
        model_kwargs = {
          "embedding_dim": 192
        },
        epochs = 69,
        training_kwargs = {
          "batch_size": 4096,
        }
  )

  hpo_result.save_to_directory('yago_ratate_hpo_ds3_num'+str(i))