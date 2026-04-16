#!/bin/bash

## Activate environments
$WITH_CONDA
source wandb-env/bin/activate

## Define the model to use
modelpath="models/geneAE_curious-totem-5139_fold0.keras"

## Define the datasets to use
datafolder="/scratch/project_465002861/rosstheo/genome_transformer_dev/data"
test_set="$datafolder/gene_sequences/test_dataset"
training_set="$datafolder/gene_sequences/training_dataset"
vfluvialis_set="$datafolder/test-species/tensorflow_datasets/vagococcus-fluvialis_dataset"
spneumoniae_set="$datafolder/test-species/tensorflow_datasets/streptococcus-pneumoniae_dataset"
efaecalis_set="$datafolder/test-species/tensorflow_datasets/enterococcus-faecalis_dataset"
ehirae_set="$datafolder/test-species/tensorflow_datasets/enterococcus-hirae_dataset"
edurans_set="$datafolder/test-species/tensorflow_datasets/enterococcus-durans_dataset"
eavium_set="$datafolder/test-species/tensorflow_datasets/enterococcus-avium_dataset"

## Run the stuff
# python evaluate_model_and_dataset.py --model-path $modelpath --dataset-path $vfluvialis_set
# python evaluate_model_and_dataset.py --model-path $modelpath --dataset-path $ehirae_set
# python evaluate_model_and_dataset.py --model-path $modelpath --dataset-path $edurans_set
# python evaluate_model_and_dataset.py --model-path $modelpath --dataset-path $eavium_set
# python evaluate_model_and_dataset.py --model-path $modelpath --dataset-path $spneumoniae_set
# python evaluate_model_and_dataset.py --model-path $modelpath --dataset-path $test_set
python evaluate_model_and_dataset.py --model-path $modelpath --dataset-path $efaecalis_set
python evaluate_model_and_dataset.py --model-path $modelpath --dataset-path $training_set

