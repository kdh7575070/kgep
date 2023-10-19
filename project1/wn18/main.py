# Pick a model
from pykeen.datasets import get_dataset
from pykeen.hpo import hpo_pipeline

dataset = get_dataset(dataset="WN18")
training = '/home/kang/.data/pykeen/datasets/wn18/wordnet-mlj12/wordnet-mlj12-train.txt'
testing =  '/home/kang/.data/pykeen/datasets/wn18/wordnet-mlj12/wordnet-mlj12-test.txt'
validation =  '/home/kang/.data/pykeen/datasets/wn18/wordnet-mlj12/wordnet-mlj12-valid.txt'

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

hpo_result.save_to_directory('wn18_ratate_hpo')