#!/bin/bash

## Activate environments
# $WITH_CONDA
# source wandb-env/bin/activate

## Define the model to use
frac_model="models/geneAE_curious-totem-5139_fold0.keras"
frc_model="models/geneAE_hopeful-violet-5138_fold0.keras"
fr_model="models/geneAE_peachy-hill-5135_fold0.keras"
rc_model="models/geneAE_glorious-silence-5133_fold0.keras"
fc_model="models/geneAE_bumbling-wind-5135_fold0.keras"
f_model="models/geneAE_wise-plasma-5134_fold0.keras"
r_model="models/geneAE_ancient-morning-5137_fold0.keras"
c_model="models/geneAE_lilac-violet-5132_fold0.keras"
models=($frac_model $frc_model $fr_model $rc_model $fc_model $f_model $r_model $c_model)

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
datasets=($test_set $training_dataset $vfluvialis_set $spneumoniae_set $efaecalis_set $ehirae_set $edurans_set $eavium_set)

## Run the stuff
# python evaluate_model_and_dataset.py --model-path $modelpath --dataset-path $vfluvialis_set
# python evaluate_model_and_dataset.py --model-path $modelpath --dataset-path $ehirae_set
# python evaluate_model_and_dataset.py --model-path $modelpath --dataset-path $edurans_set
# python evaluate_model_and_dataset.py --model-path $modelpath --dataset-path $eavium_set
# python evaluate_model_and_dataset.py --model-path $modelpath --dataset-path $spneumoniae_set
# python evaluate_model_and_dataset.py --model-path $modelpath --dataset-path $test_set
# python evaluate_model_and_dataset.py --model-path $modelpath --dataset-path $efaecalis_set
# python evaluate_model_and_dataset.py --model-path $modelpath --dataset-path $training_set

for modelpath in ${models[@]}; do
    if [ -f $modelpath ]; then
    echo "model found"
    else
    echo "$modelpath not found"
    fi
done

for datasetpath in ${datasets[@]}; do 
    if [ -f $datasetpath ]; then
    echo "model found"
    else
    echo "$datasetpath not found"
    fi
done

