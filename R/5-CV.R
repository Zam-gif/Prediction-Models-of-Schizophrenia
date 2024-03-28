library(snpStats)
geno <- read.plink('C:\\Users\\zama0\\Desktop\\6_Mind\\trainingdata')
genotype <- geno$genotypes
info <- geno$fam
extra <- geno$map
ind <- sample(5, nrow(genotype), replace = TRUE, prob = c(0.2, 0.2, 0.2, 0.2, 0.2))
dataset1 <- genotype[ind==1,]
dataset2 <- genotype[ind==2,]
dataset3 <- genotype[ind==3,]
dataset4 <- genotype[ind==4,]
dataset5 <- genotype[ind==5,]
idataset1 <- info[ind==1,]
idataset2 <- info[ind==2,]
idataset3 <- info[ind==3,]
idataset4 <- info[ind==4,]
idataset5 <- info[ind==5,]
write.plink("trainingdata\\dataset1", snps = dataset1, pedigree = idataset1[[1]], id = idataset1 [[2]], father = idataset1 [[3]], mother = idataset1 [[4]], sex = idataset1 [[5]], phenotype = idataset1 [[6]], snp.data = extra, chromosome = extra[[1]], genetic.distance = extra[[3]], position = extra[[4]], allele.1 = extra[[5]], allele.2 = extra[[6]])
write.plink("trainingdata\\dataset2", snps = dataset2, pedigree = idataset2[[1]], id = idataset2 [[2]], father = idataset2 [[3]], mother = idataset2 [[4]], sex = idataset2 [[5]], phenotype = idataset2 [[6]], snp.data = extra, chromosome = extra[[1]], genetic.distance = extra[[3]], position = extra[[4]], allele.1 = extra[[5]], allele.2 = extra[[6]])
write.plink("trainingdata\\dataset3", snps = dataset3, pedigree = idataset3[[1]], id = idataset3 [[2]], father = idataset3 [[3]], mother = idataset3 [[4]], sex = idataset3 [[5]], phenotype = idataset3 [[6]], snp.data = extra, chromosome = extra[[1]], genetic.distance = extra[[3]], position = extra[[4]], allele.1 = extra[[5]], allele.2 = extra[[6]])
write.plink("trainingdata\\dataset4", snps = dataset4, pedigree = idataset4[[1]], id = idataset4 [[2]], father = idataset4 [[3]], mother = idataset4 [[4]], sex = idataset4 [[5]], phenotype = idataset4 [[6]], snp.data = extra, chromosome = extra[[1]], genetic.distance = extra[[3]], position = extra[[4]], allele.1 = extra[[5]], allele.2 = extra[[6]])
write.plink("trainingdata\\dataset5", snps = dataset5, pedigree = idataset5[[1]], id = idataset5 [[2]], father = idataset5 [[3]], mother = idataset5 [[4]], sex = idataset5 [[5]], phenotype = idataset5 [[6]], snp.data = extra, chromosome = extra[[1]], genetic.distance = extra[[3]], position = extra[[4]], allele.1 = extra[[5]], allele.2 = extra[[6]])
