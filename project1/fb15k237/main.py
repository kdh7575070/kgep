# Pick a model
from pykeen.datasets import get_dataset
from pykeen.hpo import hpo_pipeline

training = '/home/kang/.data/pykeen/datasets/fb15k237/ds1/train.txt'
testing = '/home/kang/.data/pykeen/datasets/fb15k237/ds1/test.txt'
validation = '/home/kang/.data/pykeen/datasets/fb15k237/ds1/valid.txt'

hpo_result = hpo_pipeline(
    # n_trials = 1,  
    # training = training,
    # testing = testing,
    # validation = validation,
    # evaluator = "rankbased",
    # loss = "crossentropy",
    # negative_sampler_kwargs = {
    #   "num_negs_per_pos": 98
    # },
    # optimizer = "adam",
    # optimizer_kwargs = {
    #   "lr": 0.05869073902981315
    # },
    # model = 'ComplEx',
    # model_kwargs = {
    #   "embedding_dim": 64
    # },
    # epochs = 100,
    # training_kwargs = {
    #   "batch_size": 4096,
    # },
    
    
    n_trials = 1,  
    training = training,
    testing = testing,
    validation = validation,
    
    dataset_kwargs ={
      "create_inverse_triples": True
    },
    # evaluation_kwargs = {
      # "batch_size": None
    # },
    evaluator = "rankbased",
    evaluator_kwargs= {
      "filtered": True
    },
    filter_validation_when_testing = True,
    loss = "nssa",
    loss_kwargs = {
      "adversarial_temperature": 0.9360080942368801,
      "margin": 24.64351066224894
    },
    negative_sampler = "basic",
    negative_sampler_kwargs = {
      "num_negs_per_pos": 100
    },
    optimizer = "adam",
    optimizer_kwargs = {
      "lr": 0.001,
      "weight_decay": 0.0
    },
    model = 'rotate',
    model_kwargs = {      
      # "automatic_memory_optimization": True,
      "embedding_dim": 256
    },
    regularizer = "no",
    epochs = 101,
    training_kwargs = {
      "batch_size": 4096,
      "label_smoothing": 0.0,
    },
    training_loop = "lcwa"
)

hpo_result.save_to_directory('_ratate_hpo')

# import pathlib
# import os
# import pprint
# import pandas as pd
# import igraph as ig
# import numpy as np
# from pykeen.datasets import get_dataset
# from pykeen.hpo import hpo_pipeline

# path = '/home/kang/.data/pykeen/datasets/fb15k237/ds1/train.txt'
# table_m = pd.read_csv(path, sep='\t')
# table_m.columns = ["out", "rel", "inn"]

# table_cnt_in = table_m.groupby("inn").size().reset_index(name="n").sort_values(by="n", ascending=False)
# table_cnt_out = table_m.groupby("out").size().reset_index(name="n").sort_values(by="n", ascending=False)
# table_cnt_all = pd.merge(table_cnt_out, table_cnt_in, left_on="out", right_on="inn", how="inner")
# table_cnt_all["n"] = table_cnt_all["n_x"] + table_cnt_all["n_y"]
# table_cnt_all = table_cnt_all[["out", "n"]].rename(columns={"out": "entity"}).sort_values(by="n", ascending=False)

# thresholds = [1, 10, 100, 300, 500, 700, 800, 900, 1000, 2000, 3000, 5000]

# for i in thresholds:

#   remove_entity = []
#   for n in range(i):
#       remove_entity.append(table_cnt_all.head(i).iat[n,0])

#   # Form a new dataset

#   path = '/home/kang/.data/pykeen/datasets/fb15k237/ds1/train.txt'
#   table_m = pd.read_csv(path, sep='\t', header=None)
#   table_m.columns = ["out", "rel", "inn"]

#   for v in remove_entity:
#       table_m = table_m[table_m.out != v]
#       table_m = table_m[table_m.inn != v]

#   table_m.to_csv(path, sep='\t', index=False, header=False)

#   training = '/home/kang/.data/pykeen/datasets/fb15k237/ds1/train.txt'
#   testing = '/home/kang/.data/pykeen/datasets/fb15k237/ds1/test.txt'
#   validation = '/home/kang/.data/pykeen/datasets/fb15k237/ds1/valid.txt'

# hpo_result = hpo_pipeline(
#     n_trials = 1,  
#     training = training,
#     testing = testing,
#     validation = validation,
#     evaluator = "rankbased",
#     loss = "crossentropy",
#     negative_sampler_kwargs = {
#       "num_negs_per_pos": 98
#     },
#     optimizer = "adam",
#     optimizer_kwargs = {
#       "lr": 0.05869073902981315
#     },
#     model = 'RotatE',
#     model_kwargs = {
#       "embedding_dim": 64
#     },
#     epochs = 1000,
#     training_kwargs = {
#       "batch_size": 4096,
#     },
# )

# hpo_result.save_to_directory('fb15k237_ratate_hpo_ds3_num'+str(i))