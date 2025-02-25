import pandas as pd

def stripExt(val:str):
    return val.rsplit(".",1)[0]

## Load the raw metadata
metadata_path = "/Users/theodorross/Library/CloudStorage/OneDrive-UiTOffice365/Documents/data/e_faecium/metadata/20231026_Efm_T7_VRE_LRE_marine_metadata_Accession_numbers.xlsx"
# df = pd.read_csv(metadata_path, index_col="id")

## Load the file mapper info
file_mapper_path = "/Users/theodorross/Library/CloudStorage/OneDrive-UiTOffice365/Documents/data/e_faecium/sequence_file_mapping.csv"
file_mapper_df = pd.read_csv(file_mapper_path, index_col="ID")
file_mapper_df["File Basename"] = file_mapper_df["File Name"].apply(stripExt)
file_mapper = {str(key):str(val) for (key,val) in zip(file_mapper_df.index, file_mapper_df["File Basename"])}

# Manually add these for reasons
file_mapper["TUH2_18"] = "TUH_2_18"
file_mapper["TUH2_19"] = "TUH_2_19"


## Build the label dataframe
sheet_names = ["Tromsø 7","Kres_VRE_LRE","VRE_Study","Marine Efm"]

out_dfs = []
keepcols = ["Source","Isolation_site","Vancomycin_res","Ampicillin_res","Gentamicin_res","Linezolid_res","Year"]

for sheet in sheet_names:
    ## Load the data sheet
    df = pd.read_excel(metadata_path, sheet_name=sheet, index_col="ID")
    df.rename(columns=lambda x: x.strip(), inplace=True)
    df.index = df.index.astype(str)

    data_df = df[keepcols]
    data_df.index = data_df.index.map(file_mapper)
    out_dfs.append(data_df)

    # print(f"{sheet}")
    # print("\t", data_df.shape)
    # print("\t", data_df.index.isna().sum())
    # if data_df.index.isna().sum() > 0:
    #     print(df[data_df.index.isna()])

out_df = pd.concat(out_dfs)
out_df = out_df[out_df.index.notna()]

## Force the resistance columns to be lowercase
out_df["Vancomycin_res"] = out_df["Vancomycin_res"].str.lower()
out_df["Ampicillin_res"] = out_df["Ampicillin_res"].str.lower()
out_df["Gentamicin_res"] = out_df["Gentamicin_res"].str.lower()
out_df["Linezolid_res"] = out_df["Linezolid_res"].str.lower()

print(out_df)
out_df.to_csv("sequence-labels.csv", index=True)
