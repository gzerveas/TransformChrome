# TransformChrome

Based on:

Reference Paper: [Attend and Predict: Using Deep Attention Model to Understand Gene Regulation by Selective Attention on Chromatin](https://arxiv.org/abs/1708.00339)

BibTex Citation:
```
@inproceedings{singh2017attend,
  title={Attend and Predict: Understanding Gene Regulation by Selective Attention on Chromatin},
  author={Singh, Ritambhara and Lanchantin, Jack and Sekhon, Arshdeep  and Qi, Yanjun},
  booktitle={Advances in Neural Information Processing Systems},
  pages={6769--6779},
  year={2017}
}
```


**Feature Generation for TransformChrome/AttentiveChrome model:** 

We used the five core histone modification (listed in the paper) read counts from REMC database as input matrix. We downloaded the files from [REMC dabase](http://egg2.wustl.edu/roadmap/web_portal/processed_data.html#ChipSeq_DNaseSeq). We converted 'tagalign.gz' format to 'bam' by using the command:
```
gunzip <filename>.tagAlign.gz
bedtools bedtobam -i <filename>.tagAlign -g hg19chrom.sizes > <filename>.bam 
```
Next, we used "bedtools multicov" to get the read counts. 
Bins of length 100 base-pairs (bp) are selected from regions (+/- 5000 bp) flanking the transcription start site (TSS) of each gene. The signal value of all five selected histone modifications from REMC in bins forms input matrix X, while discretized gene expression (label /0) is the output y.

For gene expression, we used the RPKM read count files available in REMC database. We took the median of the RPKM read counts as threshold for assigning binary labels (0: gene low, 1: gene high). 

We divided the genes into 3 separate sets for training, validation and testing. It was a simple file split resulting into 6601, 6601 and 6600 genes respectively. 

We performed training and validation on the first 2 sets and then reported AUC scores of best performing epoch model for the third test data set. 

**Datasets**

We have provided a toy dataset to test out model in the data subdirectory of v2PyTorch

The complete set of 56 Cell Type datasets is located at https://zenodo.org/record/2652278

The rows are bins for all genes (100 rows per gene) and the columns are organised as follows:

GeneID, Bin ID, H3K27me3 count, H3K36me3 count, H3K4me1 count, H3K4me3 count, H3K9me3 counts, Binary Label for gene expression (0/1)  
e.g. 000003,1,4,3,0,8,4,1

**Running The Model** 

See the v2PyTorch directories to run the code.



# v2PyTorch folder includes the  TransformChrome Implementation. 
You can run it via the following command: 

```
python train.py --cell_type Toy
```


### Installation Requirements
* python>=3.5
* numpy
* pytorch-cpu
* torchvision-cpu
