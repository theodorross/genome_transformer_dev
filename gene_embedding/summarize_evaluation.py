import pandas as pd
import numpy as np





if __name__ == "__main__":

    '''
    Loop through the evaluation folders
    '''
    eval_folders = ["test"]
                    # "enterococcus-durans",
                    # "enterococcus-hirae",
                    # "enterococcus-avium",
                    # # "enterococcus-faecalis",
                    # "vagococcus-fluvialis",
                    # "streptococcus-pneumoniae"]

    for f in eval_folders:
        print(f)
        # df = pd.read_csv(f"evaluation/curious-totem-5139/{f}/cog_summary.csv")
        df = pd.read_csv(f"evaluation/curious-totem-5139/{f}/cog_categories_scores.csv")
        

    print(df)