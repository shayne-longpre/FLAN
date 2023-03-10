import flan.v2.mixtures

import os
import numpy as np
import pandas as pd
import json
import copy
import random
import seqio
from collections import defaultdict
import functools
import tensorflow as tf

dataset_name = "t0"
mix_name = dataset_name + "_zsopt"

seq_mix = seqio.get_mixture_or_task(mix_name)

dataset = seq_mix.get_dataset(
    sequence_length={"inputs": 512, "targets": 128},  # Extranous length to capture all data
    num_epochs=1,
    copy_pretokenized=True,
    # passthrough_features=["_template_idx", "_task_source", "_task_name", "_template", "_template_type"],
    # sample_fn=functools.partial(tf.data.Dataset.sample_from_datasets, stop_on_empty_dataset=True),
)



################################################################
###### Instantiate the submixtures with each template style
################################################################

# ZSOPT, FSOPT, ZSNOOPT, FSNOOPT are template styles.
# ZS means a zero-shot prompt, FS means a few-shot prompt
# OPT means the answer options for tasks with multiple choice answers are included in the template
# NOOPT means the answer options for tasks with multiple choice answers are NOT included in the template

seqio.MixtureRegistry.add(
    'cot_submix',
    tasks=[
        ('cot_zsopt', 1),    # mixing weight = 50%
        ('cot_fsopt', 1),    # mixing weight = 50%
    ])

seqio.MixtureRegistry.add(
    'dialog_submix',
    tasks=[
        ('dialog_zsopt', 1),    # mixing weight = 50%
        ('dialog_fsopt', 1),    # mixing weight = 50%
    ])

seqio.MixtureRegistry.add(
    'niv2_submix',
    tasks=[
        ('niv2_zsopt', 1),    # mixing weight = 50%
        ('niv2_fsopt', 1),    # mixing weight = 50%
    ])

seqio.MixtureRegistry.add(
    'flan2021_submix',
    tasks=[
        ('flan_zsopt', 1),      # mixing weight = 25%
        ('flan_fsopt', 1),      # mixing weight = 25%
        ('flan_zsnoopt', 1),    # mixing weight = 25%
        ('flan_fsnoopt', 1),    # mixing weight = 25%
    ])

seqio.MixtureRegistry.add(
    't0_submix',
    tasks=[
        ('t0_zsopt', 1),      # mixing weight = 25%
        ('t0_fsopt', 1),      # mixing weight = 25%
        ('t0_zsnoopt', 1),    # mixing weight = 25%
        ('t0_fsnoopt', 1),    # mixing weight = 25%
    ])

# Define the Final Flan Collection Mixture
seqio.MixtureRegistry.add(
    'flan2022_submix',
    tasks=[
        ('flan2021_submix', 0.4),     # mixing weight = 40%
        ('t0_submix', 0.32),      # mixing weight = 32%
        ('niv2_submix', 0.2),     # mixing weight = 20%
        ('cot_submix', 0.05),     # mixing weight = 5%
        ('dialog_submix', 0.03),  # mixing weight = 3%
    ])

################################################################
###### See 3 Examples of Mixtures or Submixtures you can try
################################################################
# 1. Example use cases to use just the chain-of-thought zero-shot data:
selected_mixture = seqio.get_mixture_or_task('cot_zsopt')

# 2. Example use cases to use just all chain-of-thought templates together:
# selected_mixture = seqio.get_mixture_or_task('cot_submix')

# 3. Example use cases to use the full Flan Collection:
# selected_mixture = seqio.get_mixture_or_task('flan2022_submix')

# If you're using Seqio, we suggest caching your mixture as they take a while to generate.
# If you want to read out the post-processed examples into a file, we suggest using the
# sample_fn below to collect 1 epoch of data, according to our mixing rates.
dataset = seq_mix.get_dataset(
    sequence_length={"inputs": 1024, "targets": 512},  # Extranous length to capture all data
    num_epochs=1,
    copy_pretokenized=True,
    # The passthrough features let you track the source/task/template metadata for the example
    passthrough_features=["_template_idx", "_task_source", "_task_name", "_template", "_template_type"]
    # This sample function will iterate over the mixture using the appropriate mixing rates
    # until the first dataset runs out of examples. You can run it again to get a new epoch,
    # with slightly different examples sampled.
    # sample_fn=functools.partial(tf.data.Dataset.sample_from_datasets, stop_on_empty_dataset=True),
)

# dataset.take(k).cache().repeat()