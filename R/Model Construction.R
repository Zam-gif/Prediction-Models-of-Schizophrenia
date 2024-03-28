source(“https://bioconductor.org/biocLite.R”)
biocLite(“snpStats”)
library(snpStats)
geno <- read.plink(“path to ethnicity-gender data”)
genotype <- geno$genotypes
extra <- geno$map
info <- geno$fam
ind <- sample(2, nrow(genotype), replace = TRUE, prob = c(0.8, 0.2))
traindata <- genotype[ind==1,]
testdata <- genotype[ind==2,]
itraindata <- info[ind==1,]
itestdata <- info[ind==2,]
write.plink("Training Data", snps = traindata, pedigree = itraindata[[1]], id = itraindata[[2]], father = itraindata[[3]], mother = itraindata[[4]], sex = itraindata[[5]], phenotype = itraindata[[6]], snp.data = extra, chromosome = extra[[1]], genetic.distance = extra[[3]], position = extra[[4]], allele.1 = extra[[5]], allele.2 = extra[[6]])
write.plink("Test Data", snps = testdata, pedigree = itestdata[[1]], id = itestdata [[2]], father = itestdata [[3]], mother = itestdata [[4]], sex = itestdata [[5]], phenotype = itestdata [[6]], snp.data = extra, chromosome = extra[[1]], genetic.distance = extra[[3]], position = extra[[4]], allele.1 = extra[[5]], allele.2 = extra[[6]])
