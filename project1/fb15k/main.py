# Pick a model
from pykeen.datasets import get_dataset
from pykeen.hpo import hpo_pipeline

dataset = get_dataset(dataset="FB15k")
training = dataset.training.metadata['path']
testing = dataset.testing.metadata['path']
validation = dataset.validation.metadata['path']

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

hpo_result.save_to_directory('fb15k_ratate_hpo')