import pandas as pd
import numpy as np
from Bio import Phylo
from matplotlib import pyplot as plt
from tqdm import tqdm


def get_clade_children(clade):
    ## Function to get the names of all leaves that descend from this clade
    seqs = []
    if clade.is_terminal():
        seqs.append(str(clade))
    else:
        for c in clade:
            _s = get_clade_children(c)
            seqs += _s
    return seqs


## Load the metadata
# metadata_path = "/Users/theodorross/Library/CloudStorage/OneDrive-UiTOffice365/Documents/data/e_faecium/proc-data/microreact/combined/combined-microreact-labels.csv"
# metadata_df = pd.read_csv(metadata_path, index_col="id")

## Load the tree files
# tree_path = "/Users/theodorross/Library/CloudStorage/OneDrive-UiTOffice365/Documents/data/e_faecium/proc-data/microreact/combined/combined-microreact-tree.nwk"
tree_path = "/Users/theodorross/Library/CloudStorage/OneDrive-UiTOffice365/Documents/data/e_faecium/proc-data/microreact/expanded/expanded_ml_tree_efm.tree"
# efm_tree_path = "/Users/theodorross/Library/CloudStorage/OneDrive-UiTOffice365/Documents/data/e_faecium/proc-data/microreact/combined/combined-efm-tree.nwk"
# elac_tree_path = "/Users/theodorross/Library/CloudStorage/OneDrive-UiTOffice365/Documents/data/e_faecium/proc-data/microreact/combined/combined-elac-tree.nwk"

tree = Phylo.read(tree_path, format="newick")
# efm_tree = Phylo.read(efm_tree_path, format="newick")
# elac_tree = Phylo.read(elac_tree_path, format="newick")

tree.root_at_midpoint()
# efm_tree.root_at_midpoint()
# elac_tree.root_at_midpoint()


## Test some stuff
# elac_tests = ["50966233","51021115"]
# test = elac_tree.common_ancestor(elac_tests)


## Get the internal branch length threshold
d = tree.depths()
internal_lengths = []
total_depths = []
for clade,_depth in d.items():
    if not clade.is_terminal():
        if isinstance(clade.branch_length, float):
            internal_lengths.append(clade.branch_length)
            total_depths.append(_depth)

internal_lengths = np.array(internal_lengths, dtype=float)
total_depths = np.array(total_depths, dtype=float)

n_pick = int(len(internal_lengths) * 0.05)
# print(n_pick, "/", len(internal_lengths))
idx = np.argpartition(internal_lengths, kth=-n_pick)[-n_pick:]
len_thresh = internal_lengths[idx].min()
print("Branch length minimum:", len_thresh)

idx = np.argpartition(total_depths, kth=n_pick)[n_pick:]
# print(total_depths[idx])
depth_thresh = total_depths[idx].max()
print("Clade depth maximum:", depth_thresh)


## Test each of the cuts
count = 0
test_clades = []
fig,ax = plt.subplots(4,5, figsize=(10,8))
ax = ax.ravel()
for clade,depth in tqdm(d.items()):

    if not clade.is_terminal():
        if isinstance(clade.branch_length, float):
            seqs = get_clade_children(clade)
            popfrac = len(seqs) / len(tree.get_terminals())

            # if depth >= depth_thresh:
            # if clade.branch_length >= len_thresh:
            if 0.10 < popfrac < 0.20:
                if any([_c.is_parent_of(clade) for _c in test_clades]):
                    continue

                if popfrac >= 0.00:
                    clade.color="red"
                    Phylo.draw(tree, label_func=lambda x:None, axes=ax[count], do_show=False)
                    clade.color="black"
                    ax[count].set_title(f"n={len(seqs)}/{len(tree.get_terminals())}")
                    test_clades.append(clade)
                    count += 1

print("Subsampled populations:", count)

for a in ax: a.set_axis_off()
plt.tight_layout()
plt.show()

# s0 = set(get_clade_children(test_clades[0]))
# s1 = set(get_clade_children(test_clades[1]))
# print(test_clades[1] in test_clades[0])
# print(test_clades[0].is_parent_of(test_clades[1]))


# print(np.sort(vec2[idx]))
# print(np.sort(vec2)[-n_pick:])



# fig,ax = plt.subplots(1,2)
# # poo = plt.hist(vec1, bins=50, alpha=0.5, density=True, label="total")
# _t = ax[0].hist(vec2, bins=50, alpha=0.5, density=True, label="internal")

# width = _t[1][1] - _t[1][0]
# cdf = np.cumsum(_t[0] * width)

# print(sum(cdf > .95))

# # print(len(_t[1]), len(_t[0]), len(cdf))
# ax[1].plot(_t[1][1:], cdf, label="cdf - internal")
# # plt.ylim([0,20])
# plt.legend()
# plt.show()
