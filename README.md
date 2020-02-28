DF-EAC+
=======

Information
--------

This repository contains a pyspark implementation of the algorithm described in [Scalable Fast Evolutionary k-Means Clustering by de Oliveira and Naldi (2015)](https://ieeexplore.ieee.org/abstract/document/7423998) and a modified version featuring mutation with respect to cluster centroid positions as well as cluster count.

Usage
--------

Set up your Spark cluster and launch one of the .py files with spark-submit. There are five parameters:

1. Filepath of the data. This should be purely numeric data in the standard .csv format.
2. Number of genotypes, or the number of k-means clustering algorithms running consecutively.
3. Maximum number of clusters per genotype.
4. Epsilon, or the minimum change in error required for the model to continue.
5. Mutation rate, given as a probability.
