import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import altair as alt


def convert_species_name(string, abbreviate=True):
    if abbreviate:
        words = string.split("-")
        string = " ".join([f"{string[0].capitalize()}.", words[1]])
        return string
    else:
        string = string.replace("-"," ")
        return string.capitalize()
    

def blegh(string):
    return "_".join([string[0], string.split("-")[1]])


if __name__ == "__main__":

    MODEL_NAME="curious-totem-5139"

    '''
    Summarize model performance for plotting
    '''
    # eval_folders = ["test",
    #                 "enterococcus-durans",
    #                 "enterococcus-hirae",
    #                 "enterococcus-avium",
    #                 "enterococcus-faecalis",
    #                 "vagococcus-fluvialis",
    #                 "streptococcus-pneumoniae"]
    eval_folders = ["test",
                    "enterococcus-durans",
                    "enterococcus-hirae",
                    "enterococcus-avium",
                    "enterococcus-faecalis",
                    "vagococcus-fluvialis"]

  
    ## Initialize a dictionary for storing data to plot
    plot_data = {'spec_names':[],'cog_avgs':[],
                 'gc_avgs':[], 'gc_vars':[],
                 'ani_avgs':[], 'ani_vars':[],
                 'recon_avgs':[], 'recon_vars':[],
                 'domain_avgs':[], 'domain_vars':[]}
    
    ## Loop through each dataset
    for f in eval_folders:
        # Parse the species name into something properly formatted
        if f != "test":
            plot_data["spec_names"].append(convert_species_name(f))
        else:
            plot_data["spec_names"].append("E. faecium & E. lactis")

        # Load and compute summary statistics for ANI values
        if f != "test":
            _f = "_".join([f[0], f.split("-")[1]])
        else: _f = f
        ani_df = pd.read_csv(f"../data/ANI/training-{_f}-ANI.tsv", 
                         names=['query','ref','ani','a','b'], sep='\t')
        plot_data["ani_avgs"].append(ani_df["ani"].mean())
        # plot_data["ani_avgs"].append(ani_df["ani"].median())
        plot_data["ani_vars"].append(ani_df["ani"].sem())
        # plot_data["ani_vars"].append(ani_df["ani"].std())

        # Load and compute summary statistics for COG prediction
        cog_df = pd.read_csv(f"evaluation/{MODEL_NAME}/{f}/cog_summary.csv", index_col="metric")
        plot_data["cog_avgs"].append(cog_df.loc["genes completely correct","percent"])

        # Load and compute summary statistics for reconstruction performance
        recon_df = pd.read_csv(f"evaluation/{MODEL_NAME}/{f}/reconstruction_data.csv")
        plot_data["recon_avgs"].append((recon_df["recon_acc"]*100).mean())
        # plot_data["recon_avgs"].append((recon_df["recon_acc"]*100).median())
        plot_data["recon_vars"].append((recon_df["recon_acc"]*100).sem())
        plot_data["gc_avgs"].append((recon_df['gc_frac']*100).mean())
        # plot_data["gc_avgs"].append((recon_df['gc_frac']*100).median())
        plot_data["gc_vars"].append((recon_df['gc_frac']*100).sem())
        # plot_data["recon_vars"].append((recon_df["recon_acc"]*100).std())

        # Load and compute summary statistics for domain clustering
        domain_df = pd.read_csv(f"evaluation/{MODEL_NAME}/{f}/domain_clustering_summary.csv")
        plot_data["domain_avgs"].append((domain_df["frac"]*100).mean())
        # plot_data["domain_avgs"].append((domain_df["frac"]*100).median())
        plot_data["domain_vars"].append((domain_df["frac"]*100).sem())
        # plot_data["domain_vars"].append((domain_df["frac"]*100).std())
    
    ## Create melted dataframes for the average and standard error values of each statistic
    plot_df = pd.DataFrame(plot_data)
    m1 = plot_df.melt(id_vars=["spec_names","ani_avgs", "ani_vars", 'gc_avgs', 'gc_vars'], 
                      value_vars=["cog_avgs",'recon_avgs','domain_avgs'],
                      var_name="task", value_name='avgs')
    m2  = plot_df.melt(id_vars=["spec_names","ani_avgs", "ani_vars", 'gc_avgs', 'gc_vars'], 
                      value_vars=['recon_vars','domain_vars'],
                      var_name='task', value_name='errs')
    
    # Format some text values
    m1["task"] = m1["task"].str.replace("_avgs","")
    m2["task"] = m2["task"].str.replace("_vars","")

    # Merge the dataframes into a single plotting frame
    plot_df = pd.merge(m1, m2, how='outer')
    print(plot_df)

    # Format the task names for figure displays
    header_mapper = {"cog":'Perfect COG prediction rate (%)',
                     'domain':'Domain cluster purity (%)',
                     'recon':'Reconstruction Accuracy'}
    plot_df["task"] = plot_df['task'].map(header_mapper)

    ## Define a base for plotting, computing minimum and maximum values for error bars
    base = alt.Chart(plot_df).transform_calculate(
        ymin="datum.avgs-datum.errs",
        ymax="datum.avgs+datum.errs",
        ani_min="datum.ani_avgs-datum.ani_vars",
        ani_max="datum.ani_avgs+datum.ani_vars",
        gc_min="datum.gc_avgs-datum.gc_vars",
        gc_max="datum.gc_avgs+datum.gc_vars"
    )

    ## Plot the of each value
    ani_scatter = base.mark_point(filled=True).encode(
        x = alt.X('ani_avgs:Q', title="ANI with training data (%)").scale(zero=False),
        y = alt.Y('avgs:Q', title=None).scale(zero=False),
        color = alt.Color("spec_names:N", title="Species")
    )
    gc_scatter = base.mark_point(filled=True).encode(
        x = alt.X('gc_avgs:Q', title="GC content (%)").scale(zero=False),
        y = alt.Y('avgs:Q', title=None).scale(zero=False),
        color = alt.Color("spec_names:N", title="Species")
    )

    ## Plot the errorbars
    ani_hbars = base.mark_errorbar().encode(
        x = alt.X('ani_min:Q', title="ANI with training data (%)").scale(zero=False),
        x2 = alt.X2('ani_max:Q'),
        y = alt.Y('avgs:Q', title=None).scale(zero=False),
        color = alt.Color("spec_names:N", title="Species")
    )
    gc_hbars = base.mark_errorbar().encode(
        x = alt.X('gc_min:Q', title="GC content (%)").scale(zero=False),
        x2 = alt.X2('gc_max:Q'),
        y = alt.Y('avgs:Q', title=None).scale(zero=False),
        color = alt.Color("spec_names:N", title="Species")
    )

    ani_vbars = base.mark_errorbar().encode(
        x = alt.X('ani_avgs:Q', title="ANI with training data (%)").scale(zero=False),
        y = alt.Y('ymin:Q', title=None).scale(zero=False),
        y2 = alt.Y2('ymax:Q'),
        color = alt.Color("spec_names:N", title="Species")
    )
    gc_vbars = base.mark_errorbar().encode(
        # x = alt.X('ani_avgs:Q', title="ANI with training data (%)").scale(zero=False),
        x = alt.X('gc_avgs:Q', title="GC content (%)").scale(zero=False),
        y = alt.Y('ymin:Q', title=None).scale(zero=False),
        y2 = alt.Y2('ymax:Q'),
        color = alt.Color("spec_names:N", title="Species")
    )

    # outplot = scatter + errbars
    ## Face and format the plots
    ani_outplot = (ani_scatter + ani_hbars + ani_vbars).facet(
        column = alt.Column("task:N", title=None)
    ).resolve_scale(
        x="independent",
        y="independent"
    ).configure_legend(
        labelFontStyle="italic"
    )
    gc_outplot = (gc_scatter + gc_hbars + gc_vbars).facet(
        column = alt.Column("task:N", title=None)
    ).resolve_scale(
        x="independent",
        y="independent"
    ).configure_legend(
        labelFontStyle="italic"
    )

    ani_outplot.save(f"evaluation/{MODEL_NAME}/ani_summary_plot.png")
    gc_outplot.save(f"evaluation/{MODEL_NAME}/gc_summary_plot.png")
