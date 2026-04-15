import pandas as pd
import numpy as np





if __name__ == "__main__":

    '''
    Loop through the evaluation folders
    '''
    eval_folders = ["test",
                    "enterococcus-durans",
                    "enterococcus-hirae",
                    "enterococcus-avium",
                    # "enterococcus-faecalis",
                    "vagococcus-fluvialis",
                    "streptococcus-pneumoniae"]

    for f in eval_folders:
        print()
        print(f)
        # df = pd.read_csv(f"evaluation/curious-totem-5139/{f}/cog_summary.csv")

        df = pd.read_csv(f"evaluation/curious-totem-5139/{f}/domain_clustering_summary.csv")
        print('clustering purity:', df['frac'].mean())
        print('clustering purity:', df['frac'].median())

        df = pd.read_csv(f"evaluation/curious-totem-5139/{f}/reconstruction_data.csv")
        print('reconstruction accuracy:', df['recon_acc'].mean())
        print('reconstruction accuracy:', df['recon_acc'].median())

        df = pd.read_csv(f"evaluation/curious-totem-5139/{f}/cog_summary.csv", index_col="Unnamed: 0")
        print("cogs completely correct:", df.loc["genes with labels",'percent'])
        

    # print(df)