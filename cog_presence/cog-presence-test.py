import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling1D, Dense
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.metrics import AUC, Precision, Recall

# import wandb
# from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

from utils.ContigPA_LinearEncoding import ContigPA_LinearEncoding
from utils.AssemblyTransformer import TransformerBlock
from utils.ContigPA_Generator import ContigPA_Generator



if __name__ == "__main__":

    '''Parse the input arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", default="Springfield", type=str, help="Hardware platform used for job training", required=True)
    parser.add_argument("--embedding-dim", "-z", default=1024, type=int, help="Dimensionality of the contig embedding vectors.", required=False)
    parser.add_argument("--label-column", "-y", default="Vancomycin_res", type=str, help="Data column to use as labels for training.", required=False)
    parser.add_argument("--batch-size", "-b", default=4, type=int, help="Batch size for training.", required=False)
    parser.add_argument("--max-contigs", "-c", default=100, type=int, help="Maximum contigs allowed in a sequence for training.", required=False)
    parser.add_argument("--shuffle-contigs", default=True, type=bool, help="Switch to control whether or not the order of contigs in a sequence are shuffled.", required=False)
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate for the optimizer.", required=False)
    parser.add_argument("--patience", default=25, type=int, help="Patience for early stopping.", required=False)
    parser.add_argument("--val-split", default=0.2, type=float, help="Fraction of the training data to use for validation", required=False)
    args = parser.parse_args()

    print("Looking for GPUs...")
    print(tf.config.list_physical_devices('GPU'))


    '''Initialize wandb'''
    # wandb.init(
    #     project="genome-transformer",
        
    #     config={
    #         "gene-representation":"binary P/A",
    #         "gene-embedding-type":"none",
    #         "contig-representation":"binary gene P/A vector",
    #         "contig-embedding-type":"linear",
    #         "embedding-dim":args.embedding_dim,
    #         "label-column":args.label_column,
    #         "batch-size":args.batch_size,
    #         "max-contigs":args.max_contigs,
    #         "shuffle-contigs":args.shuffle_contigs,
    #         "learning-rate":args.lr,
    #         "patience":args.patience,
    #         "val-split":args.val_split,
    #         "platform":args.platform
    #     }, 
        
    #     sync_tensorboard=True
    # )

    '''Define data directories'''
    print("Defining data directories")
    if args.platform == "local":
        contig_path = "../data/contig-gene-presence-absence.csv"
        labels_path = "../data/sequence-labels.csv"
        splits_path = "../data/sequence-splits.csv"

    elif args.platform == "Springfield":
        contig_path = "/storage/data/e_faecium/contig-gene-presence-absence.csv"
        labels_path = "/storage/data/e_faecium/sequence-labels.csv"
        splits_path = "/storage/data/e_faecium/sequence-splits.csv"

    elif args.platform == "LUMI":
        contig_path = "/scratch/project_465000263/rosstheo/genome_transformer_dev/data/contig-gene-presence-absence.csv"
        labels_path = "/scratch/project_465000263/rosstheo/genome_transformer_dev/data/sequence-labels.csv"
        splits_path = "/scratch/project_465000263/rosstheo/genome_transformer_dev/data/sequence-splits.csv"


    '''Load data'''
    ## Load the data splits information
    print("Loading data splits...")
    splits_df = pd.read_csv(splits_path, index_col="ID")
    training_seqs = splits_df[splits_df["split"] == "training"].index.tolist()
    validation_seqs = splits_df[splits_df["split"] == "validation"].index.tolist()

    ## Load and process the contig data
    print("Loading contig data...")
    # contig_df = pd.read_csv("../data/contig-gene-presence-absence.csv", nrows=1e3)    ## For debugging
    contig_df = pd.read_csv(contig_path, low_memory=False)                              ## For running
    contig_df["sequence"] = contig_df["sequence"].astype('string')
    sequences = contig_df["sequence"].unique()
    contig_df.set_index(["sequence","scaffold"], inplace=True)
    genes = contig_df.columns.to_numpy()
    
    ## Load and process the labels
    print("Loading labels...")
    labels = pd.read_csv(labels_path, index_col="ID")[args.label_column]
    # labels = labels.loc[sequences]      ## for debugging
    # labels.loc["2016_199"] = "yes"      ## for debugging (randomly chosen)
    
    # If the labels are resistance data
    if "res" in args.label_column:
        label_mapper = {"no":0, "yes":1}

    # Apply the label mapper and convert the dataframe to one-hot encoding
    labels = labels.map(label_mapper)
    one_hot_labels = to_categorical(labels, num_classes=len(label_mapper))
    labels = pd.DataFrame({key:one_hot_labels[:,ix] for ix,key in enumerate(label_mapper.keys())},
                          index = labels.index)

    # training_seqs = [s for s in training_seqs if s in sequences]        ## for debugging
    # validation_seqs = [s for s in validation_seqs if s in sequences]    ## for debugging

    ## Define the data generator
    print("Defining data generators...")
    training_generator = ContigPA_Generator(contig_df.loc[training_seqs], labels.loc[training_seqs],
                                            batch_size=args.batch_size, max_contigs=args.max_contigs, 
                                            shuffle_contigs=args.shuffle_contigs)
    val_generator = ContigPA_Generator(contig_df.loc[validation_seqs], labels.loc[validation_seqs], 
                                       batch_size=args.batch_size, max_contigs=args.max_contigs, 
                                       shuffle_contigs=False)
    

    '''Define model hyperparameters'''
    z_dim = args.embedding_dim
    x_dim = len(genes)


    '''Define the model'''
    print("Defining model...")
    ## Define the embedding and transformer layers
    embedding_layer = ContigPA_LinearEncoding(input_dim=x_dim, embedding_dim=z_dim, use_bias=False)
    transformer_layer = TransformerBlock(z_dim, z_dim, num_heads=5, dropout_rate=0.1, key_dim=255)


    ## Define the feed model
    inp = Input(shape=(None, x_dim))
    emb = embedding_layer(inp)
    t_out = transformer_layer(emb)
    temp_pool = GlobalAveragePooling1D()(t_out)
    d1 = Dense(256, activation="relu")(temp_pool)
    d2 = Dense(len(label_mapper), activation="softmax")(d1)

    model = Model(inputs=[inp], outputs=[d2])

    model_metrics = ["accuracy", AUC(), Precision(), Recall()]
    model.compile(optimizer=Adam(learning_rate=args.lr), loss="categorical_crossentropy", 
                  metrics=model_metrics)


    '''Define callbacks'''
    print("Define callbacks...")
    ## Wandb callbacks
    # wandb_logger = WandbMetricsLogger()
    # wandb_chkpt = WandbModelCheckpoint("models")

    ## Early stopping callback
    earlystop = EarlyStopping(monitor="val_loss", patience=args.patience)


    '''Test it out'''
    print("Train model...")
    H = model.fit(x=training_generator, validation_data=val_generator, epochs=100, 
                  callbacks=[earlystop])
    # H = model.fit(x=training_generator, validation_data=val_generator, epochs=10, 
    #               callbacks=[earlystop, wandb_logger, wandb_chkpt])
    print(H.history)

    # test_seqs,test_labels = data_generator[0]
    # print(test_seqs.shape, test_labels.shape)
    # test_out = model.predict(test_seqs)
    # print(test_out)
    print("Done.")
