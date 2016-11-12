##############################################################
#
#
# Genetic algorithm classes
#
#
##############################################################
import sys
import math
import time
import random
import networkx as nx
import matplotlib.pyplot as plt
from Tkinter import *

class Population:
	def __init__(self, graph, populationSize, crossoverFunc, mutationFunc):
		self.graph = graph
		self.size = populationSize
		self.crossoverFunc = crossoverFunc
		self.mutationFunc = mutationFunc
		self.cuts = []

	def getInitPopulation(self):
		# generate population_size members
		for i in range(self.size):
			cut = []

			# generate 0 or 1 for every node in graph
			for i in range(len(self.graph.nodes())):
				cut.append(random.randint(0, 1))

			# add new cut to population's cut
			self.cuts.append(cut)

	# (source: http://stackoverflow.com/questions/10324015/fitness-proportionate-selection-roulette-wheel-selection-in-python)
	def parentSelect(self):
		fitness_list = []

		# create list of fitnesses
		for cut in self.cuts:
			fitness_list.append(self.getFitness(cut)

		# sum up fitnesses
		fitness_sum = 0
		for value in fitness_list:
			fitness_sum = fitness_sum + value

		# find random number between 0 and fitness sum
		prob = random.randint(0, fitness_sum)

		# search until current value > fitness sum
		current_value = 0
		for i in range(self.size):
			current_value = current_value + fitness_list[i]
			if(current_value > prob):
				return self.cuts[i]

	# generate new population
	def breedNewGeneration(self):
		pass
	## put code shit

	## put code shit
	def sortByFitness(self):
		self.cuts.sort(None, self.getFitness, True)

	def getFitness(self, cut):
		graph = self.graph
		fitness = 0
		for i in range(0, len(cut)):
			if(cut[i] == 0):
				city = str(i)
				neighbors = graph.neighbors(city)
				for neighbor in neighbors:
					j = int(neighbor)
					if(cut[j] == 1):
						fitness += graph.edge[city][neighbor]['weight']
		return fitness

	##def crossover(cut1, cut2):

# function that returns a networkx graph fully initialized based on an input file
def initializeGraph(filepath):
	# open file
	cityfile = open(filepath, "r") # open the file for reading

	# the file parsing comes in two stages
	# 1. read in the nodes and their coordinates
	# 2. read in the edges in the graph
	cityGraph = nx.Graph()
	for line in cityfile:
		splitLine = line.split(" ")

		if len(splitLine) == 3:
			# initialize a cities object containing each city and its coordinates
			cityGraph.add_node(splitLine[0], x=float(splitLine[1]), y=float(splitLine[2]))

		if len(splitLine) == 2:
			splitLine[0] = str(int(splitLine[0]))
			splitLine[1] = str(int(splitLine[1]))
			edgeWeight = math.hypot(cityGraph.node[splitLine[0]]['x'] - cityGraph.node[splitLine[1]]['x'], cityGraph.node[splitLine[0]]['y'] - cityGraph.node[splitLine[1]]['y'])
			cityGraph.add_edge(splitLine[0], splitLine[1], weight=edgeWeight)

	cityfile.close()
	return cityGraph
def blank():
	return
def main():
	inputFile = sys.argv[1]
	cityGraph = initializeGraph(inputFile)

	pop = Population(cityGraph, 10, blank, blank)
	pop.getInitPopulation()
	print pop.getFitness(pop.cuts[0])
	print pop.getFitness(pop.cuts[-1])

	pop.sortByFitness()
	print pop.getFitness(pop.cuts[0])
	print pop.getFitness(pop.cuts[-1])

if __name__ == "__main__":
    main()
