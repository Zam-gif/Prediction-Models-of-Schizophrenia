# SNP-based Prediction of Schizophrenia Using Machine Learning

## Introduction

Welcome to our research repository, where we demonstrate the feasibility of creating powerful predictive models for schizophrenia based on Genome-Wide Association (GWA) studies, which is feasible through the appropriate selection of machine learning rules.

Our objective is to harness the power of machine learning to craft SNP-based predictive models tailored to ethnicity-gender groups. These include separate models for European-American females (EA-F), European-American males (EA-M), African-American females (AA-F), and African-American males (AA-M). Each model is meticulously developed to reflect the unique genetic and demographic characteristics of its target population.

In this repository, you will find a comprehensive suite of resources that demonstrate the feasibility and effectiveness of deploying predictive models for schizophrenia. 

## GWAS with PLINK
[PLINK](http://zzz.bwh.harvard.edu/plink/) is a highly recognized software tool for GWAS data analysis developed by [Shaun Purcell](http://zzz.bwh.harvard.edu/) at Harvard, MGH, and the Broad Institute.
PLINK 1.7 was used for the analysis

### Data Loading 
The Genetic Association Information Network (GAIN) data on schizophrenia were stored in the PLINK format, which is commonly used for exchanging genotype/phenotype data and is compatible with most GWAS software tools.
PLINK formats are fully documented [here](http://zzz.bwh.harvard.edu/plink/data.shtml).

### Data Processing
Our data processing was designed to ensure the quality of the data for machine learning analysis. We applied quality control filters. 

The steps include:
1) Individuals were filtered into separate groups of males and females, and the analysis for each group was conducted separately.
   
PLINK Commands:
```sh
plink --bfile \pathwaytofiles(bed, bim, fam) --filter-males --make-bed --out \pathwaytonewfiles
plink --bfile \pathwaytofiles(bed, bim, fam) --filter-females --make-bed --out \pathwaytonewfiles
```
2) Individuals with missing phenotypes were removed
   
PLINK Command:
```sh
plink --bfile \pathwaytofiles(bed, bim, fam) --prune --make-bed --out \pathwaytonewfiles
```
3) Genetic variants that failed the Hardy-Weinberg test with a significance threshold p< 0.0001 were removed
   
PLINK Command:
```sh
plink --bfile \pathwaytofiles(bed, bim, fam) --hwe 0.0001 --make-bed --out \pathwaytonewfiles
```
4) Genetic variants with a genotyping rate of less than 90% were removed
   
PLINK Command:
```sh
plink --bfile \pathwaytofiles(bed, bim, fam) --geno 0.1 --make-bed --out \pathwaytonewfiles
```
5) Genetic variants with minor allele frequency (MAF) less than 10% were removed
   
PLINK Command:
```sh
plink --bfile \pathwaytofiles(bed, bim, fam) --maf 0.1 --make-bed --out \pathwaytonewfiles
```
6) Individuals with too much missing genotype data (more than 10%) were removed
   
PLINK Command:
```sh
plink --bfile \pathwaytofiles(bed, bim, fam) --mind 0.1 --make-bed --out \pathwaytonewfiles
```
Population specifications after preprocessing are presented in Table S3 in the S1 File.

### Model construction
 After cleaning the data, we used R code to randomly split it into two sets: 80% for training and 20% for testing. R code is given in the folder R (see Model construction)
 
R code:
```sh
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
```

Table 1 in the article displays the number of individuals from each ethnicity-gender group used for training and testing purposes.

### 5-fold CV
1)	Then, within each group, we randomly divided the training datasets into 5 folds by using R code. R code is given in the folder R (see 5-fold CV).
   
R code:
```sh
library(snpStats)
geno <- read.plink("path to trainingdata")
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
```

2)	To constitute the training dataset, we merge 4 different folds for feature selection and prognostic model construction and set aside one fold for validation. That step was performed five times to obtain five different combinations
   
PLINK commands: 
```sh
plink --file \pathwaytorecodeddataset1 --merge-list \pathwaytothedocument_2345.txt --recode --out \pathwaytothemergeddata_2345
plink --file \pathwaytorecodeddataset2 --merge-list \pathwaytothedocument_1345.txt --recode --out \pathwaytothemergeddata_1345
plink --file \pathwaytorecodeddataset3 --merge-list \pathwaytothedocument_1245.txt --recode --out \pathwaytothemergeddata_1245
plink --file \pathwaytorecodeddataset4 --merge-list \pathwaytothedocument_1235.txt --recode --out \pathwaytothemergeddata_1245
plink --file \pathwaytorecodeddataset5 --merge-list \pathwaytothedocument_1234.txt --recode --out \pathwaytothemergeddata_1235
```
3)	For feature (SNP) selection, we employed the Cochran-Mantel-Haenszel (CMH) association test for these 4 folds on each ethnicity-gender-specific training set to rank the SNPs according to their association with schizophrenia, and selected and recorded those SNPs with a Benjamini-Hochberg False Discover Rate (BH-FDR) less than T where T was treated as a hyperparameter of the learning rule—T was assumed to vary in {0.01, 0.05, 0.1}. So that step was repeated five times and obtained five different combinations.

PLINK Commands:
```sh
plink --file \pathwaytomergeddataset__2345(bed, bim, fam) --assoc --adjust --out \pathwaytoanalyzedfiles
plink --file \pathwaytomergeddataset__1345(bed, bim, fam) --assoc --adjust --out \pathwaytoanalyzedfiles
plink --file \pathwaytomergeddataset__1245(bed, bim, fam) --assoc --adjust --out \pathwaytoanalyzedfiles
plink --file \pathwaytomergeddataset__1235(bed, bim, fam) --assoc --adjust --out \pathwaytoanalyzedfiles
plink --file \pathwaytomergeddataset__1234(bed, bim, fam) --assoc --adjust --out \pathwaytoanalyzedfiles
```
4)	Significantly associated SNPs were recorded in the document
   
5)	Not significantly associated SNPs were removed from both training and test datasets.

To do this, we used the PLINK command:
  
```sh
plink --bfile \pathwaytotrainingdata(bed, bim, fam) --extract \pathwaytothedocumentfromstep5.txt --make-bed --out \pathwaytofinalizedtrainingdata
plink --bfile \pathwaytotestdata(bed, bim, fam) --extract \pathwaytothedocumentfromstep3.txt --make-bed --out \pathwaytofinalizedtestdata
```

6)	Finalized training and test data were recoded to ped/map 

PLINK Commands:
```sh
plink --bfile \pathwaytofinalizedtrainingdata(bed, bim, fam) --recode --out \pathwaytorecodedtrainingdata
plink --bfile \pathwaytofinalizedtestdata(bed, bim, fam) --recode --out \pathwaytorecodedtestdata
```

7)	Next, we performed Naive Bayes, TAN, Random Forest, and Logistic Regression model selection by applying stratified 5-CV on each ethnicity-gender training dataset. The AUC and accuracy of each classifier were recorded, and the overall accuracy was calculated. Next, we train and build the models around this and evaluate the final model using the test data. This part is done in Python script which is given in the folder Py.

8)	For the Tree-Augmented Naive Bayes (TAN) model, we used WEKA tool. The TAN model based on the training dataset was constructed and evaluated on the test dataset. The data (ped/map) was recoded to *arff (WEKA) format via the TRES tool.  AUC and accuracies were recorded.

WEKA:
```sh
weka.classifiers.BayesNet.TAN –S ENTROPY 
```


