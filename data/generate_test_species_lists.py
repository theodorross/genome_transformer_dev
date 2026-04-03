import os


def generate_species_lists(species):
    ## Generate a list of file locations for a given species
    tack_path = "ncbi_dataset/ncbi_dataset/data"
    src_path = f"{source_path}/{species}/{tack_path}"
    sample_list = next(os.walk(src_path))[1]

    ## Eggnog and HMMER needs proteins, fwd pass needs CDS
    proteins_tsv = ""
    cds_tsv = ""

    ## Store the path to each desired datafile
    for s in sample_list:
        protein_file = f"{src_path}/{s}/protein.faa"
        cds_file = f"{src_path}/{s}/cds_from_genomic.fna"

        if os.path.exists(protein_file) and os.path.exists(cds_file): 
            proteins_tsv += f"{s}\t{protein_file}\n"
            cds_tsv += f"{s}\t{cds_file}\n"

    ## Save the tsv files
    with open(f"test-species/{species}/proteins.tsv","w") as f:
        f.write(proteins_tsv)

    with open(f"test-species/{species}/cds.tsv","w") as f:
        f.write(cds_tsv)


if __name__=="__main__":
    source_path = "/Users/tro119/Library/CloudStorage/OneDrive-UiTOffice365/Documents/data/reference_genomes"

    generate_species_lists('enterococcus-avium')
    generate_species_lists('enterococcus-durans')
    generate_species_lists('enterococcus-faecalis')
    generate_species_lists('enterococcus-hirae')
    generate_species_lists('streptococcus-pneumoniae')
    generate_species_lists('vagococcus-fluvialis')
