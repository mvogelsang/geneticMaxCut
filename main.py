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
#
# class Population:
# 	def __init__(self, graph, population_size, crossover_func):
# 		self.graph = graph
# 		self.population_size = population_size
# 		self.crossover_func = crossover_func
#
# 	# generate random initial population
# 	def get_init_population(self):
# 	# put code shit
#
# 	# select parent for breeding
# 	def parent_select(self):
# 	## put code shit
#
# 	# generate new population
# 	def breed_new_generation(self):
# 	## put code shit
#
# 	# sort population
# 	def sort_pop(self):
# 	## put code shit
#
# 	# find fittest member of population
# 	def get_fittest(self):
# 	## put code shit
#
# 	##def crossover(cut1, cut2):
# 	## FILL IN LATER
#
# class Cut:
# 	def __init__(self, cut, mutate_func):
# 		self.cut = cut
# 		self.fitness = self.get_fitness()
# 		self.mutate_func = mutate_func
#
# 	# get fitness for stored cut
# 	def get_fitness(self):
# 	## put code shit
#
# 	##def mutate(self):
# 	## FILL IN LATER

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
	
def main():
	inputFile = sys.argv[1]
	cityGraph = initializeGraph(inputFile)

if __name__ == "__main__":
    main()
