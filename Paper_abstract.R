# tokenize all the available abstracts from Ecography in ISI-WEB
root <- "~/Desktop/A/Me/Papers_abstracts"
journal <- "ecography.txt"
Sys.setlocale("LC_ALL", "C")
eco <- read.delim(file.path(root, journal), h=T, sep = "\t")
eco[608,]
Sys.setlocale("LC_ALL", "C")