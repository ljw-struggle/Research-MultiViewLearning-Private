library(ggseqlogo)
library(ggplot2)
library(repr)


# $ mkdir -p ./_figure/motif_logo

file_list <- list.files(path = "./_doc/human_HOCOMO/", full.names = TRUE)

for (file in file_list) {
    filename <- basename(file)
    name_without_ext <- strsplit(filename, "_")[[1]][1]

    lines <- readLines(file)
    start_line <- grep("letter-probability matrix:", lines) + 1
    matrix_lines <- lines[start_line:length(lines)]
    matrix <- matrix(NA, nrow = length(matrix_lines), ncol = 4)
    for (i in 1:seq_along(matrix_lines)) {
        row_values <- unlist(strsplit(matrix_lines[i], "\\s+"))
        row_values <- as.numeric(row_values)
        matrix[i, ] <- row_values
    }
    # print(matrix)
    frequency_matrix <- as.matrix(t(matrix))
    rownames(frequency_matrix) <- c("A", "C", "G", "T")
    # print(frequency_matrix)


    # list_col_schemes(v = T)
    # p1 <- ggseqlogo(frequency_matrix, method="prob", col_scheme="base_pairing")
    p2 <- ggseqlogo(frequency_matrix, method = "bits", col_scheme = "nucleotide")

    # add the axis
    p2 <- p2 + theme(axis.line = element_line(colour = "black"))

    # add the white background
    # p2 <- p2 + theme(panel.background = element_rect(fill = "white"))

    # save the plot
    output_path <- paste0("./_figure/motif_logo/", name_without_ext)
    output_path <- paste0(output_path, ".pdf")
    ggsave(output_path, p2, width = 10, height = 5, dpi = 300)
}



# lines <- readLines("./_doc/human_HOCOMO/AHR_HUMAN.H10MO.B.meme")
# start_line <- grep("letter-probability matrix:", lines) + 1
# matrix_lines <- lines[start_line:length(lines)]
# matrix <- matrix(NA, nrow = length(matrix_lines), ncol = 4)
# for (i in 1:seq_along(matrix_lines)) {
#     row_values <- unlist(strsplit(matrix_lines[i], "\\s+"))
#     row_values <- as.numeric(row_values)
#     matrix[i, ] <- row_values
# }
# # print(matrix)
# frequency_matrix <- as.matrix(t(matrix))
# rownames(frequency_matrix) <- c("A", "C", "G", "T")
# # print(frequency_matrix)


# # list_col_schemes(v = T)
# # p1 <- ggseqlogo(frequency_matrix, method="prob", col_scheme="base_pairing")
# p2 <- ggseqlogo(frequency_matrix, method = "bits", col_scheme = "nucleotide")

# # add the axis
# p2 <- p2 + theme(axis.line = element_line(colour = "black"))

# # add the white background
# # p2 <- p2 + theme(panel.background = element_rect(fill = "white"))

# # save the plot
# # ggsave("./_figure/motif_logo/" + name_without_ext + ".pdf", p2, width = 10, height = 5, dpi = 300)


# options(repr.plot.width = 10, repr.plot.height = 3)
# print(p2)
