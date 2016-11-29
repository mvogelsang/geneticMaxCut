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


trialNumber = 0

class Population:
	def __init__(self, graph, populationSize, crossoverFunc, mutationFunc, mutationRate):
		self.graph = graph
		self.size = populationSize
		self.crossover = crossoverFunc
		self.mutation = mutationFunc
		self.cuts = []
		self.mutationRate = mutationRate

	def getInitPopulation(self):
		# generate population_size members
		for i in range(self.size):
			cut = []

			# generate 0 or 1 for every node in graph
			for i in range(len(self.graph.nodes())):
				cut.append(random.randint(0, 1))

			# add new cut to population's cut
			self.cuts.append(cut)

	def parentSelect(self):
		p1 = random.choice(self.cuts)
		p2 = random.choice(self.cuts)

		if(self.getFitness(p1) > self.getFitness(p2)):
			return p1
		else:
			return p2

	# generate new population
	def breedNewGeneration(self):
		new_pop = []
		new_pop.append(self.cuts[0])

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

		# breed best no matter what
		new_pop.append(self.crossover(self.cuts[0], self.parentSelect()))

		# breed best and worst matter what
		new_pop.append(self.crossover(self.cuts[0], self.cuts[-1]))

		# run mutation on potential population
		i = 1
		while i < len(new_pop):
			cut = new_pop[i]
			if(random.random() < self.mutationRate):
				self.mutation(cut)
			i += 1

		# put the new pop into the true population
		self.cuts = new_pop

		# sort potential population
		self.sortByFitness()

		# keep only the best population size cuts
		self.cuts = self.cuts[:self.size]

	def sortByFitness(self):
		self.cuts.sort(None, self.getFitness, True)

	def getFitness(self, cut):
		graph = self.graph
		fitness = 0

		# iterate through each node in the graph
		for i in range(0, len(cut)):
			# if in cut 0, find all connected nodes
			if(cut[i] == 0):
				city = str(i)
				neighbors = graph.neighbors(city)
				# find all neigbors in cut 1
				for neighbor in neighbors:
					j = int(neighbor)
					# add weight of edge if it crosses the cut
					if(cut[j] == 1):
						fitness += graph.edge[city][neighbor]['weight']
						# fitness += 1
		return fitness

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

		if len(splitLine) == 2:
			numCities = splitLine[0]
			i = 1
			while i <= numCities:
				cityGraph.add_node(str(i))

		if len(splitLine) == 3:
			splitLine[0] = str(int(splitLine[0]))
			splitLine[1] = str(int(splitLine[1]))
			edgeWeight = int(splitLine[3])
			cityGraph.add_edge(splitLine[0], splitLine[1], weight=edgeWeight)

	cityfile.close()
	return cityGraph

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
	f = plt.figure()

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
	#child = parent1[:randBegin] + parent2[randBegin:randEnd] + parent1[randEnd:]

	child = parent1[:]
	child[randBegin:randEnd] = parent2[randBegin:randEnd]

	if len(child) != len(parent1):
		print 'ERR!'

	return child

def crossover2(parent1, parent2):
	child = []

	i = 0
	# continue until len(child) == len(parent)
	while i < len(parent1):
		# random bit if parent bits are different
		if(parent1[i] != parent2[i]):
			newBit = 0

			# newBit is 1 50% of the time
			if(random.random() > .5):
				newBit = 1

			child.append(newBit)
		# same bit as parents if parent bits are same
		else:
			child.append(parent1[i])

		i += 1

	if len(child) != len(parent1):
		print 'ERR!'

	return child

def mutation1(citizen):
	# calculate number of flips to perform
	numFlips = random.randint(0, len(citizen)-1)

	i = 0
	# perform specified flips by XORing value at randLocation
	while i < numFlips:
		randLocation = random.randint(0, len(citizen)-1)
		citizen[randLocation] = citizen[randLocation] ^ 1
		i += 1

def mutation2(citizen):
	# get random location
	randLocation = random.randint(0, len(citizen) - 1)

	# flip value at randLocation by XOR
	citizen[randLocation] = citizen[randLocation] ^ 1

def runGeneticAlgorithm(outFile, graph, pop, numGenerations):
	fitnessList = []
	i = 0

	pop.getInitPopulation()
	pop.sortByFitness()

	fitnessList.append(pop.getFitness(pop.cuts[0]))

	while i < 10 or i < minGenerations or float(fitnessList[-1] - fitnessList[-10])/float(fitnessList[-10])*100 < .0001:
		pop.breedNewGeneration()
		fitnessList.append(pop.getFitness(pop.cuts[0]))
		i += 1

	illustrateCut(graph, 'Best Final Gen Cut', pop.cuts[0])
	fitnessOverTime(outFile, fitnessList)

def aggregate(graph, pop):
	cutList = pop.cuts
	membersToTake = int(math.ceil(float(len(cutList))*.3))
	cutList = cutList[:membersToTake]
	voteList=[]

	i=0
	while i < len(graph.nodes()):
		voteList.append({'0':0, '1': 0})
		i += 1

	for cut in cutList:
		if cut[0] == 0:
			flipBits(cut)

		i=0
		while i < len(graph.nodes()):
			voteList[i][str(cut[i])] += 1
			i += 1

	wocSoln = []
	tieTracker = []
	i=0
	while i < len(graph.nodes()):
		if voteList[i]['0'] > voteList[i]['1']:
			wocSoln.append(0)
		else:
			if voteList[i]['0'] == voteList[i]['1']:
				wocSoln.append(0)
				tieTracker.append(i)
			else:
				wocSoln.append(1)
		i += 1

	for i in tieTracker:
		originalScore = pop.getFitness(wocSoln)
		wocSoln[i]=1
		newScore = pop.getFitness(wocSoln)
		if originalScore > newScore:
			wocSoln[i] = 0

	return pop.getFitness(wocSoln), wocSoln

def flipBits(cut):
	for bit in cut:
		if bit == 0:
			bit = 1
		else:
			bit = 0

def fitnessOverTime(fitnessList):
	f = plt.figure()

	genNum = []
	for i in range(len(fitnessList)):
		genNum.append(i)

	plt.title('Gen v. Fitness')
	plt.plot(genNum, fitnessList, 'bo',genNum, fitnessList, 'k')

	plt.axis([0,genNum.pop() + 55, 0, fitnessList[-1] + 500])
	plt.show()

def main():
	plt.ion()

	random.seed()

	inputFile = sys.argv[1]
	runType = sys.argv[2]
	crossover = sys.argv[3]
	mutation = sys.argv[4]

	cityGraph = initializeGraph(inputFile)

	if(crossover == '1'):
		crossover = crossover1
	else:
		crossover = crossover2

	if(mutation == '1'):
		mutation = mutation1
	else:
		mutation = mutation2

	for i in range(3):
		pop = Population(cityGraph, 100, crossover, mutation, .1)
		runGeneticAlgorithm(outFile, cityGraph, pop, 200)

		trialNumber += 1

	plt.ioff()
    plt.show()
if __name__ == "__main__":
    main()
