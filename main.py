import os
import sys
import ipdb
import pandas as pd
import diskcache as dc
from absl import app, flags
from gemba.utils import get_gemba_scores


flags.DEFINE_string("method", "GEMBA-MQM", "Which method to use?")
flags.DEFINE_string("model", "gpt-4", "OpenAI model")
flags.DEFINE_string("source", None, "Filepath to the source file.")
flags.DEFINE_string("hypothesis", None, "Filepath to the translation file.")
flags.DEFINE_string("source_lang", None, "Source language name.")
flags.DEFINE_string("target_lang", None, "Target language name.")
flags.DEFINE_bool(
    "load_batch",
    False,
    "True if uploading the batches in GEMBA-BATCH mode.",
)
flags.DEFINE_list(
    "batch_ids",
    None,
    "list of the batch_ids that should be downloaded and processed",
)
flags.DEFINE_string(
    "dataset_name",
    None,
    "The name of the dataset that should be processed for the batches",
)
flags.DEFINE_bool(
    "download_batches",
    False,
    "if False then the local_path flag is used a path to the local batches. If True the batches will be downloaded using the dataset_name and the number of batches given in batch_ids",
)
flags.DEFINE_string(
    "local_path",
    None,
    "path to batch files",
)


def main(argv):
    FLAGS = flags.FLAGS
    assert FLAGS.source is not None, "Source file must be provided."
    assert FLAGS.hypothesis is not None, "Hypothesis file must be provided."

    # check that source and hypothesis files exists
    if not os.path.isfile(FLAGS.source):
        print(f"Source file {FLAGS.source} does not exist.")
        sys.exit(1)
    if not os.path.isfile(FLAGS.hypothesis):
        print(f"Hypothesis file {FLAGS.hypothesis} does not exist.")
        sys.exit(1)

    assert FLAGS.source_lang is not None, "Source language name must be provided."
    assert FLAGS.target_lang is not None, "Target language name must be provided."

    # load both files and strip them
    with open(FLAGS.source, "r") as f:
        source = f.readlines()
    source = [x.strip() for x in source]
    with open(FLAGS.hypothesis, "r") as f:
        hypothesis = f.readlines()
    hypothesis = [x.strip() for x in hypothesis]

    assert len(source) == len(
        hypothesis
    ), "Source and hypothesis files must have the same number of lines."

    answers = get_gemba_scores(
        source,
        hypothesis,
        FLAGS.source_lang,
        FLAGS.target_lang,
        FLAGS.method,
        FLAGS.model,
        FLAGS.load_batch,
        FLAGS.batch_ids,
        FLAGS.dataset_name,
        FLAGS.download_batches,
        FLAGS.local_path,
    )

    for answer in answers:
        print(answer)


if __name__ == "__main__":
    app.run(main)
