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

# the following lines supress an unnecessary and unavoidable warning in the
# command line. The warning comes inherently from calling certain animation functions
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

class Population:
	def __init__(self, graph, populationSize, crossoverFunc, mutationFunc):
		self.graph = graph
		self.size = populationSize
		self.crossover = crossoverFunc
		self.mutation = mutationFunc
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
			fitness_list.append(self.getFitness(cut))

		# sum up fitnesses
		fitness_sum = 0
		for value in fitness_list:
			fitness_sum = fitness_sum + value

		# find random number between 0 and fitness sum
		prob = random.randint(0, int(fitness_sum))

		# search until current value > fitness sum
		current_value = 0
		for i in range(self.size):
			current_value = current_value + fitness_list[i]
			if(current_value >= prob):
				return self.cuts[i]

	# generate new population
	def breedNewGeneration(self):
		# keep best 10% from last generation (assume already sorted)
		# num_to_keep = int(math.floor(self.size * .1))
		num_to_keep = 2
		new_pop = self.cuts[:num_to_keep]

		# breed population size times
		for i in range(self.size):
			# get parents
			parent1 = self.parentSelect()
			parent2 = self.parentSelect()

			# get 2 children based on parents
			child1 = self.crossover(parent1, parent2)
			child2 = self.crossover(parent1, parent2)

			# force mutate if children in pop already
			if(child1 in new_pop):
				self.mutation(child1)
			if(child2 in new_pop):
				self.mutation(child2)

			# add new children to potential population
			new_pop.append(child1)
			new_pop.append(child2)

		# put the new pop into the true population
		self.cuts = new_pop

		# sort potential population
		self.sortByFitness()

		# keep only the best population size cuts
		self.cuts = self.cuts[:self.size]

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
						# fitness += graph.edge[city][neighbor]['weight']
						fitness += 1
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

def illustrateFullGraph(graph):
	# force nodes to render according to their x,y position and not randomly
	pos = {}
	for node in graph.nodes():
	    pos[node] = [graph.node[node]["x"], graph.node[node]["y"]]

	plt.figure()
	plt.title('Full Graph')
	nx.draw_networkx_labels(graph, pos)
	nx.draw_networkx_nodes(graph, pos, node_color=(.7,.7,.7))
	nx.draw_networkx_edges(graph, pos, edgelist=graph.edges(), edge_color='grey')
	plt.pause(.1)

def illustrateCut(graph, cutTitle, cut):
	# draw in a new window
	plt.figure()

	# set the title
	plt.title(cutTitle)

	# force nodes to render according to their x,y position and not randomly
	pos = {}
	for node in graph.nodes():
	    pos[node] = [graph.node[node]["x"], graph.node[node]["y"]]

	# get the two separate sets of nodes
	s1 = []
	s2 = []
	for i in range(0, len(cut)):
		if( cut[i] == 0 ):
			s1.append(str(i))
		else:
			s2.append(str(i))
	nx.draw_networkx_labels(graph, pos)
	nx.draw_networkx_nodes(graph, pos, nodelist=s1, node_color='red')
	nx.draw_networkx_nodes(graph, pos, nodelist=s2, node_color='blue')

	# lightly animate all of the edges in the graph
	nx.draw_networkx_edges(graph, pos, edgelist=graph.edges(), edge_color='grey', width=.2)

	# get all of the edges in the cut
	cutEdges=[]
	for i in range(0, len(cut)):
		if(cut[i] == 0):
			city = str(i)
			neighbors = graph.neighbors(city)
			for neighbor in neighbors:
				j = int(neighbor)
				if(cut[j] == 1):
					cutEdges.append( (str(i),str(j)) )

	nx.draw_networkx_edges(graph, pos, edgelist=cutEdges, edge_color='black', width=1.4)
	plt.pause(.1)

def crossover1(parent1, parent2):
	middle = int(math.floor((len(parent1)-1)/2))
	end = len(parent1)-1

	# get the boundary for a random chunk from path 2
	randBegin = random.randint(0,middle)
	randEnd = random.randint(middle+1, end)

	#chop out pieces of parent1 and parent2 and make the child
	child = parent1[:randBegin] + parent2[randBegin:randEnd] + parent1[randEnd:]

	if len(child) != len(parent1):
		print 'ERR!'

	return child

def mutation1(citizen):
	numFlips = random.randint(0, len(citizen)-1)

	i = 0
	while i < numFlips:
		randLocation = random.randint(0, len(citizen)-1)
		citizen[randLocation] = citizen[randLocation] ^ 1
		i += 1

def main():
	inputFile = sys.argv[1]
	cityGraph = initializeGraph(inputFile)

	# turn on pyplot's interactive mode to allow live updating of the graph
	plt.ion()

	illustrateFullGraph(cityGraph)

	pop = Population(cityGraph, 50, crossover1, mutation1)

	pop.getInitPopulation()
	pop.sortByFitness()
	illustrateCut(cityGraph, 'Best Initial Cut', pop.cuts[0])
	print pop.getFitness(pop.cuts[0])

	for i in range(50):
		pop.breedNewGeneration()
		print pop.getFitness(pop.cuts[0])

	pop.sortByFitness()
	illustrateCut(cityGraph, 'Best Final Gen Cut', pop.cuts[0])
	print pop.getFitness(pop.cuts[0])


	# keep the graphs up at the end
	plt.ioff()
	plt.show()
if __name__ == "__main__":
    main()
