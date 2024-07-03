### Perform enrichment analysis on the gene expression profiles of drugA treatment
# 0. settings
rm(list = ls())
library(clusterProfiler)
library(org.Hs.eg.db)
library(usefun)

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
absolute_path <- normalizePath(script_path)
project_dir <- get_parent_dir(get_parent_dir(get_parent_dir(get_parent_dir(absolute_path))))

# convert_ratio for ratio and FC calculate
convert_ratio <- function(ratio_string) {
  parts <- strsplit(ratio_string, "/")[[1]]
  numerator <- as.numeric(parts[1])
  denominator <- as.numeric(parts[2])
  return(numerator / denominator)
}


enrichKEGG_expre_changes <- function(env_name, drugA_index){

    input_file_name = paste(env_name, "_drugA_", as.character(drugA_index), "_expression.csv", sep="")
    output_file_name = paste(env_name, "_drugA_", as.character(drugA_index), "_enrichment.csv", sep="")

    drugA_expression_data = read.csv(file.path(project_dir, "scripts", "downstream_analysis", "working_log", input_file_name), row.names = 1, header = TRUE, check.names = FALSE)[1:10,]
    res = list()
    for(i in 1:10)
    {
      expression_change_data = drugA_expression_data[i,]
      intersect_genes = colnames(expression_change_data)[abs(expression_change_data)>1]
      
      ego <- enrichKEGG(gene          = intersect_genes,
                        universe      = colnames(expression_change_data),
                        organism      = "hsa",
                        pAdjustMethod = "BH",
                        pvalueCutoff  = 1,
                        qvalueCutoff  = 0.05)
      res[[i]] = as.data.frame(ego)

      if(dim(res[[i]])[2] == 0){
        res[[i]] = null_ego
      } 
      res[[i]]$GeneRatioNum <- unlist(lapply(res[[i]]$GeneRatio, convert_ratio))
      res[[i]]$BgRatioNum <- unlist(lapply(res[[i]]$BgRatio, convert_ratio))
      res[[i]]$FC = res[[i]]$GeneRatioNum/res[[i]]$BgRatioNum
      res[[i]] = res[[i]][, c("ID", "Description", "FC")]
    }
    
    #
    pathways_vector <- unique(unlist(lapply(res, rownames)))
    total_df <- data.frame(matrix(0, nrow = length(pathways_vector), ncol = total_step))
    colnames(total_df) <- paste0("FC", 1:total_step)


    pathway_id_description_df = do.call(rbind, res)
    rownames(total_df) <- pathway_id_description_df[pathways_vector,"Description"]
    
    for(i in 1:length(res)){
      total_df[, i] = res[[i]][pathways_vector,]$FC
    }
    total_df[is.na(total_df)] <- 0.0
    out_file = paste(file.path(project_dir, "scripts", "downstream_analysis", "working_log", output_file_name), sep="")
    write.table(total_df, file=out_file, sep="\t", row.names=T, col.names=T)
    print("Enrichment analysis successed!")
}

args <- commandArgs(trailingOnly = TRUE)
env_name = as.character(args[1])
drugA_index = as.numeric(args[2])

total_step = 10

# for null_ego
null_ego <- data.frame(
  ID = character(0),
  Description = character(0),
  GeneRatio = character(0),
  BgRatio = character(0),
  pvalue = numeric(0),
  p.adjust = numeric(0),
  qvalue = numeric(0),
  geneID = character(0),
  Count = integer(0)
)


enrichKEGG_expre_changes(env_name, drugA_index)
