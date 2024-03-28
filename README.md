# SNP-based Prediction of Schizophrenia Using Machine Learning

#### Author: Zamart Ramazanova

## Introduction
Welcome to our research repository, where we demonstrate the feasibility of creating powerful predictive models for schizophrenia based on Genome-Wide Association (GWA) studies, which is feasible through the appropriate selection of machine learning rules.

Our objective is to harness the power of machine learning to craft SNP-based predictive models tailored to ethnicity-gender groups. These include separate models for European-American females (EA-F), European-American males (EA-M), African-American females (AA-F), and African-American males (AA-M). Each model is meticulously developed to reflect the unique genetic and demographic characteristics of its target population.

In this repository, you will find a comprehensive suite of resources that demonstrate the feasibility and effectiveness of deploying predictive models for schizophrenia. 

## GWAS with PLINK

PLINK is a highly recognized software tool for GWAS data analysis developed by Shaun Purcell at Harvard, MGH, and the Broad Institute.

PLINK 1.7 was used for the analysis.

## Data Processing

Our data processing was designed to ensure the quality of the data for machine learning analysis. The steps include:

1) Individuals were filtered into separate groups of males and females. The analysis for each group was done separately.
   
 __PLINK Commands:__
   
plink --bfile \pathwaytofiles(bed, bim, fam) --filter-males --make-bed --out \pathwaytonewfiles
plink --bfile \pathwaytofiles(bed, bim, fam) --filter-females --make-bed --out \pathwaytonewfiles

2) Individuals with missing phenotypes were removed.
   
__PLINK Command:__

plink --bfile \pathwaytofiles(bed, bim, fam) --prune --make-bed --out \pathwaytonewfiles

3) Genetic variants that failed the Hardy-Weinberg test with a significance threshold of 0.0001 were removed
PLINK Command:
plink --bfile \pathwaytofiles(bed, bim, fam) --hwe 0.0001 --make-bed --out \pathwaytonewfiles

5) Genetic variants with a genotyping rate of less than 90% were removed
PLINK Command:
plink --bfile \pathwaytofiles(bed, bim, fam) --geno 0.1 --make-bed --out \pathwaytonewfiles
