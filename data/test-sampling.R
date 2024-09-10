library(ggtree)
library(tidytree)
library(phangorn)
library(ggplot2)
rm(list=ls())

## Load the metadata
metadata_path = "/Users/theodorross/Library/CloudStorage/OneDrive-UiTOffice365/Documents/data/e_faecium/proc-data/microreact/combined/combined-microreact-labels.csv"
df = read.csv(metadata_path, row.names="id")

## Load the trees
tree_path = "/Users/theodorross/Library/CloudStorage/OneDrive-UiTOffice365/Documents/data/e_faecium/proc-data/microreact/combined/combined-microreact-tree.nwk"
tree = read.tree(tree_path)

g = ggtree(tree) + theme_tree()
print(g)


# efm_tree_path = "/Users/theodorross/Library/CloudStorage/OneDrive-UiTOffice365/Documents/data/e_faecium/proc-data/microreact/combined/combined-efm-tree.nwk"
# elac_tree_path = "/Users/theodorross/Library/CloudStorage/OneDrive-UiTOffice365/Documents/data/e_faecium/proc-data/microreact/combined/combined-elac-tree.nwk"
# 
# efm.tree = midpoint(read.tree(efm_tree_path))
# elac.tree = midpoint(read.tree(elac_tree_path))
# 
# g = ggtree(efm.tree) + theme_tree()
# print(g)


# n.cuts = c()
# thresh = seq(0,0.1,0.001)
# 
# for (t in thresh){
#   n.cuts = c(n.cuts, sum(tree$edge.length > t) )
# }
# 
# df = data.frame(cutoff=thresh, cuts=n.cuts)
# 
# g = ggplot(data=df, mapping=aes(x=cutoff, y=cuts)) + geom_line()
# print(g)


## Define the test sequences
# test.seqs = c()
# for (f in list.files("holdout-seqs")){
#   s = scan(paste("holdout-seqs",f, sep="/"), character())
#   test.seqs = c(test.seqs, s)
# }
# 
# df[,"split"] = "training"
# df[test.seqs,"split"] = "test"



# sample.lineages = function(tree, target.frac){
#   target.num = round(length(elac.tree$tip.label) * target.frac)
#   
#   print(target.num)
# }
# 
# sample.lineages(elac.tree, 0.19)
# 
# print(length(elac.tree$tip.label) * 0.19)
# ## Test something out
# # g = ggtree(elac.tree) + geom_nodelab(aes(label = node), hjust = -0.5)
# # print(g)
# 
# test.df = data.frame()
# temp = as_tibble(elac.tree)
# for (n in 1:dim(temp)[1]){
#   tryCatch({
#       subset <- tree_subset(elac.tree, n)
#       if (length(subset$tip.label) < 35){
#         # print(c(n, length(subset$tip.label)))
#         test.df[n,"n.seqs"] = length(subset$tip.label)
#         test.df[n,"seqs"] = subset$tip.label
#       }
#     },
#     error = function(cond){},
#     warning = function(cont){},
#     finally = {}
#   )
# }
# test.df = na.omit(test.df)
# 
# 
# # g = ggtree(elac.tree) + geom_highlight(node=68, fill="steelblue", alpha=0.7)
# # print(g)