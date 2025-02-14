### Enrichment analysis of drug A gene expression profiles
# 0. Settings
rm(list = ls())  # Clear the environment
library(clusterProfiler)
library(org.Hs.eg.db)

# Convert ratio string into numeric ratio (e.g., "2/5" -> 0.4)
convert_ratio <- function(ratio_string) {
  parts <- strsplit(ratio_string, "/")[[1]]
  numerator <- as.numeric(parts[1])
  denominator <- as.numeric(parts[2])
  return(numerator / denominator)
}

# Function to perform KEGG enrichment analysis for transcriptomic changes
enrichKEGG_expre_changes <- function(cell_line, drugA_index, drugA_expression_data_path, output_path) {
    # Load drug A expression data
    drugA_expression_data <- read.csv(
        paste(drugA_expression_data_path, cell_line, "_drugA_", as.character(drugA_index), ".csv", sep = ""),
        row.names = 1,
        header = TRUE,
        check.names = FALSE
    )[1:10, ]
    
    res <- list()
    for (i in 1:10) {
        # Extract expression change data for step i
        expression_change_data <- drugA_expression_data[i, ]
        # Identify intersecting genes with absolute expression changes > 1
        intersect_genes <- colnames(expression_change_data)[abs(expression_change_data) > 1]
        
        # Perform KEGG enrichment analysis
        ego <- enrichKEGG(
            gene          = intersect_genes,
            universe      = colnames(expression_change_data),
            organism      = "hsa",
            pAdjustMethod = "BH",
            pvalueCutoff  = 1,
            qvalueCutoff  = 0.05
        )
        res[[i]] <- as.data.frame(ego)

        # Handle empty results
        if (dim(res[[i]])[2] == 0) {
            res[[i]] <- null_ego
        }

        # Calculate fold change (FC)
        res[[i]]$GeneRatioNum <- unlist(lapply(res[[i]]$GeneRatio, convert_ratio))
        res[[i]]$BgRatioNum <- unlist(lapply(res[[i]]$BgRatio, convert_ratio))
        res[[i]]$FC <- res[[i]]$GeneRatioNum / res[[i]]$BgRatioNum
        res[[i]] <- res[[i]][, c("ID", "Description", "FC")]
    }
    
    # Combine results for all steps
    pathways_vector <- unique(unlist(lapply(res, rownames)))
    total_df <- data.frame(matrix(0, nrow = length(pathways_vector), ncol = total_step))
    colnames(total_df) <- paste0("FC", 1:total_step)

    pathway_id_description_df <- do.call(rbind, res)
    rownames(total_df) <- pathway_id_description_df[pathways_vector, "Description"]
    
    for (i in 1:length(res)) {
        total_df[, i] <- res[[i]][pathways_vector, ]$FC
    }
    total_df[is.na(total_df)] <- 0.0
    
    # Save results to output file
    out_file <- paste(output_path, cell_line, "_drugA_", as.character(drugA_index), ".csv", sep = "")
    write.table(total_df, file = out_file, sep = "\t", row.names = TRUE, col.names = TRUE)
}

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
cell_line <- as.character(args[1])
total_res_path <- as.character(args[2])  # Path to the cell line-specific result file
null_ego_expression_path <- as.character(args[3])  # Path to the null drug A expression data
drugA_expression_data_path <- as.character(args[4])  # Path to drug A expression data
output_path <- as.character(args[5])  # Path to save enrichment results

# Read total results and extract unique drug A indices
temp <- read.csv(total_res_path, header = TRUE, check.names = FALSE)
drugA_index_arr <- as.vector(unique(temp["first_ind"]))$first_ind
total_step <- 10

# Prepare null enrichment results for empty cases
drugA_expression_data <- read.csv(null_ego_expression_path, row.names = 1, header = TRUE, check.names = FALSE)[1:10, ]
expression_change_data <- drugA_expression_data[2, ]
intersect_genes <- colnames(expression_change_data)[abs(expression_change_data) > 1]

ego <- enrichKEGG(
    gene          = intersect_genes,
    universe      = colnames(expression_change_data),
    organism      = "hsa",
    pAdjustMethod = "BH",
    pvalueCutoff  = 1,
    qvalueCutoff  = 0.05
)
null_ego <- as.data.frame(ego)

# Perform enrichment analysis for each drug A index
cnt <- 0
print(length(drugA_index_arr))
for (drugA_index in drugA_index_arr) {
    start_time <- Sys.time()
    print(cnt)
    cnt <- cnt + 1
    enrichKEGG_expre_changes(cell_line, drugA_index, drugA_expression_data_path, output_path)
    end_time <- Sys.time()
    runtime <- end_time - start_time
    cat("Program runtime:", runtime, "seconds\n")
}
