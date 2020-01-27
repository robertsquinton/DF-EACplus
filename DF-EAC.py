#DELETE ALL THE IMPORTS YOU AREN'T USING
import sys
import itertools
from math import sqrt
from operator import add
from os.path import join, isfile, dirname
import numpy as np

from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel

def parseData(line):
	"""
	Takes a line of numeric data seperated by commas, outputs as a vector
	"""
	vector = tuple(line.split(','))
	vector = tuple([float(number) for number in vector])
	return vector

def calculateSurvival(scores):
	"""
	Given a vector of scores, generates a probability distribution biased in favor of lower scores, then returns a vector of samples from that distribution
	Example: Scores are [4, 7, 12, 5], so the distribution is [.4, .2, .1, .3], the dice are rolled, and the return vector turns out to be [#0, #3, #0, #1]
	"""
	scores_copy = list(scores)
	probability_distribution = list(scores)
	current_share = len(scores)
	total_shares = sum(range(len(scores) + 1))
	
	while(current_share > 0):
		probability_distribution[np.argmin(scores_copy)] = (float(current_share) / float(total_shares))
		scores_copy[np.argmin(scores_copy)] = np.inf
		current_share -= 1

	survivors = np.random.choice(a = len(scores_copy), size = len(scores_copy), p = probability_distribution)
	return(survivors)

def kMutate(geno, max_k, mutation_rate):
	"""
	Given a list of genotypes, perform mutations on the number of clusters within each genotype.
	The genotypes can either delete a random cluster or split a random cluster into two, which reduces or increases the number of clusters by 1.
	"""
	mutate_choices = np.random.random(len(geno)) < mutation_rate
	out_geno = []
	
	for j in range(len(geno)):
		centroids = geno[j].clusterCenters

		if mutate_choices[j] == True:
			discriminant = np.random.random()
			if ((discriminant < .5) & (len(centroids) > 2)) | (len(centroids) >= max_k):	#Delete cluster
				del(centroids[np.random.choice(len(centroids))])
			elif ((discriminant >= .5) & (len(centroids))) | (len(centroids) <= 2):		#Split cluster
				centroids.append(centroids[np.random.choice(len(centroids))])

		out_geno.append(KMeansModel(centroids))

	return(out_geno)

if __name__ == "__main__":
	if (len(sys.argv) != 6):
		print("Usage: /path/to/spark/bin/spark-submit --driver-memory 2g " + \
		"DF-EAC.py (data filepath) (number of genotypes) (max K) (epsilon) (mutation rate)")
		sys.exit(1)

	conf = SparkConf() \
		.setAppName("DF-EAC") \
		.set("spark.executor.memory", "2g")
	sc = SparkContext(conf=conf)

#input should be purely numeric data with columns seperated with commas, and rows seperated with newlines
	data_fp = sys.argv[1]
	data_rdd = sc.textFile(data_fp).map(parseData)
	train_rdd = data_rdd.sample(False, .8)
	valid_rdd = data_rdd.subtract(train_rdd)

	g = int(sys.argv[2])
	genotype = [None]*g

	max_k = int(sys.argv[3])
	genotype_k = np.random.randint(2, max_k + 1, size = g)

	epsilon = float(sys.argv[4])
	genotype_score = [None]*g
	m_rate = float(sys.argv[5])
	generation = 0

	while True:
		for i in range(g):
			model = KMeans.train(train_rdd, k = genotype_k[i], maxIterations = 6, epsilon = 0.0, initialModel = genotype[i]) 
			genotype[i] = KMeansModel(model.clusterCenters)
			genotype_score[i] = model.computeCost(valid_rdd)

		print('Score mean: %f standard dev: %f' % (np.mean(genotype_score), np.std(genotype_score)))
		print(genotype_k)

		average_score = np.mean(genotype_score)

		generation += 1
		print("Generation %d completed" % generation)

		if generation > 1:
			if abs((average_score - last_average_score)/last_average_score) < epsilon:
				out = ''
				for i in genotype[np.argmin(genotype_score)].clusterCenters:
					out += (','.join([str(j) for j in i]) +'\n')
				with open('DF-EAC_model.csv', 'w') as file:
					file.write(out)
				break

		new_generation = [None] * g
		new_generation_k = [None] * g
		new_generation_index = calculateSurvival(genotype_score)
		
		for i in range(g):
			new_generation[i] = genotype[new_generation_index[i]]

		new_generation = kMutate(new_generation, max_k, m_rate)
		new_generation_k = [len(x.clusterCenters) for x in new_generation]

		last_average_score = average_score
		genotype = new_generation
		genotype_k = new_generation_k
