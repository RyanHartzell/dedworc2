# This runs! But is pretty slow. At any rate, the final result is interesting, and probably applicable if we can modify the scenario and VMAS task for our shit

from benchmarl.algorithms import MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

experiment = Experiment(
   task=VmasTask.SAMPLING.get_from_yaml(),
   algorithm_config=MappoConfig.get_from_yaml(),
   model_config=MlpConfig.get_from_yaml(),
   critic_model_config=MlpConfig.get_from_yaml(),
   seed=0,
   config=ExperimentConfig.get_from_yaml(),
)
experiment.run()