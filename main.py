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
import os
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

	cityGraph = nx.Graph()
	for line in cityfile:
		splitLine = line.split()

		if len(splitLine) == 2:
			numCities = int(splitLine[0])
			i = 0
			while i < numCities:
				cityGraph.add_node(str(i))
				i += 1

		if len(splitLine) == 3:
			splitLine[0] = str(int(splitLine[0])-1)
			splitLine[1] = str(int(splitLine[1])-1)
			edgeWeight = int(splitLine[2])
			cityGraph.add_edge(splitLine[0], splitLine[1], weight=edgeWeight)

	cityfile.close()
	return cityGraph

def illustrateFullGraph(graph):
	pos = nx.circular_layout(graph)


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

	pos = nx.circular_layout(graph)

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
	# perform specified flips at randLocation
	while i < numFlips:
		randLocation = random.randint(0, len(citizen)-1)
		citizen[randLocation] = flipOfBit(citizen[randLocation])
		i += 1

def mutation2(citizen):
	# get random location
	randLocation = random.randint(0, len(citizen) - 1)

	# flip value at randLocation
	citizen[randLocation] = flipOfBit(citizen[randLocation])

def runGeneticAlgorithm( graph, pop, minGenerations):
	fitnessList = []
	i = 0

	pop.getInitPopulation()
	pop.sortByFitness()

	fitnessList.append(pop.getFitness(pop.cuts[0]))

	while i < 10 or i < minGenerations or float(fitnessList[-1] - fitnessList[-10])/float(fitnessList[-10])*100 > .00001:
		pop.breedNewGeneration()
		fitnessList.append(pop.getFitness(pop.cuts[0]))
		i += 1
		# print fitnessList[-1]

	return fitnessList


def moreOnesThanZeroes(cut):
	ones = 0
	zeroes = 0
	for bit in cut:
		if bit == 0:
			zeroes += 1
		else:
			ones += 1

	if ones > zeroes:
		return True
	else:
		return False

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
		if cut[0] ==1:
			flipBits(cut)
		if moreOnesThanZeroes(cut):
			flipBits(cut)

		i=0
		while i < len(graph.nodes()):
			voteList[i][str(cut[i])] += 1
			i += 1

	wocSoln = []
	for i in range(graph.number_of_nodes()):
		wocSoln.append(-1)

	for edge in graph.edges():
		graph.edge[edge[0]][edge[1]]['popularity'] = 0

	for cut in cutList:
		alterEdgePopularityByCut(graph, cut)

	edgeList = graph.edges()
	edgeList.sort(key = lambda x: graph[x[0]][x[1]]['popularity'], reverse=True)

	for i,edge in enumerate(edgeList):
		if graph.edge[edge[0]][edge[1]]['popularity'] == 0:
			endOfUseful = i
			break
	edgeList = edgeList[0:endOfUseful]

	for i,edge in enumerate(edgeList):
		putEdgeIntoCut( pop, wocSoln, voteList, edge)

	for i in range(len(wocSoln)):
		wocSoln[i]=0
		zeroScore = pop.getFitness(wocSoln)
		wocSoln[i]=1
		oneScore = pop.getFitness(wocSoln)
		if zeroScore > oneScore:
			wocSoln[i] = 0
		else:
			wocSoln[i] = 1

	return wocSoln

def putEdgeIntoCut( pop, cut, voteList, edge):
	indexA = int(edge[0])
	indexB = int(edge[1])
	placementA = alreadyPlaced(cut[indexA])
	placementB = alreadyPlaced(cut[indexB])
	if(placementA and not placementB):
		bitToPlace = cut[indexA]
		cut[indexB] = flipOfBit(bitToPlace)
		return

	if(not placementA and placementB):
		bitToPlace = cut[indexB]
		cut[indexA] = flipOfBit(bitToPlace)
		return

	if(not placementA and not placementB):
		votesForA0 = voteList[indexA]['0']
		votesForB0 = voteList[indexB]['0']
		if(votesForA0 > votesForB0):
			cut[indexA] = 0
			cut[indexB] = 1
			return

		if(votesForA0 < votesForB0):
			cut[indexA] = 1
			cut[indexB] = 0
			return

		if(votesForA0 == votesForB0):
			choice1 = cut[:]
			choice1[indexA] = 0
			choice1[indexB] = 1

			choice2 = cut[:]
			choice2[indexA] = 1
			choice2[indexB] = 0

			fit1 = pop.getFitness(choice1)
			fit2 = pop.getFitness(choice2)

			if(fit1 >= fit2):
				cut = choice1
				return
			else:
				cut = choice2
				return
	if(placementA and placementB):
		return


def alreadyPlaced(setIdentifier):
	if setIdentifier == 0 or setIdentifier == 1:
		return True
	else:
		return False

def alterEdgePopularityByCut(graph, cut):
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
						graph.edge[city][neighbor]['popularity'] += 1

def flipBits(cut):
	for i,bit in enumerate(cut):
		cut[i] = flipOfBit(bit)

def flipOfBit(bit):
	if bit == 0:
		return 1
	else:
		return 0

def fitnessOverTime(fitnessList, inputPath, crossover, mutation, iteration):
	pathArr = inputPath.split('/')
	filename = pathArr[-1]
	category = pathArr[-2]
	runCombo = 'c'+str(crossover)+'m'+str(mutation)
	outPath = './results/'+ runCombo + '/charts/' + category + '_' + filename +'_' + str(iteration)+ '.png'
	optimal = getOptimalDist(category, filename)

	plt.ion()
	f = plt.figure()

	genNum = []
	for i in range(len(fitnessList)):
		genNum.append(i)

	plt.title('Fitness v. Generation\n' + 'Crossover: ' + str(crossover) + ' Mutation: ' + str(mutation) + ' Trial: ' + str(iteration) )
	plt.plot(genNum, fitnessList, '.75', genNum, fitnessList, 'bx')
	plt.plot((genNum[0], (genNum[-1]*1.1+1)), (optimal, optimal), 'r-')
	plt.axis([0,(genNum[-1]*1.1+1), 0, fitnessList[-1]*1.25])
	plt.draw()
	plt.pause(.0001)

	f.savefig(outPath)
	plt.close(f)

def writeData(inPath, crossover, mutation, avgGeneticDist, avgGeneticTime, avgWocDist, avgWocTime, numNodes, numEdges):
	pathArr = inPath.split('/')
	filename = pathArr[-1]
	category = pathArr[-2]
	runCombo = 'c'+str(crossover)+'m'+str(mutation)

	outPath = './results/'+ runCombo + '/' + category + '.res'

	optimal = getOptimalDist(category, filename)
	line  = filename + ' ' + str(numNodes) + ' ' + str(numEdges) + ' ' + str(optimal) + ' ' + str(avgGeneticDist) + ' ' + str(avgGeneticTime) + ' ' + str(avgWocDist) + ' ' + str(avgWocTime) + '\n'
	f = open(outPath, 'a+')
	f.write(line)
	f.close()

def getOptimalDist(category, filename):
	infoPath = './data/optimalValues/' + category + '.opt'
	f = open(infoPath, 'r')
	for line in f:
		if filename in line:
			data = line.split()
			optimal = int(data[-1])
			break
	f.close()
	return optimal

def getAvg(numList):
	total = 0.0
	for num in numList:
		total += float(num)
	avg = total/float(len(numList))
	return avg

def main():

	inputPath = ''
	minGenerations = 20
	numtrials = 3
	crossoverChoice = sys.argv[1]
	mutationChoice = sys.argv[2]
	rudyFiles = os.listdir('./data/rudy')
	isingFiles = os.listdir('./data/ising')
	rudyFiles.sort() # sorts normally by alphabetical order
	rudyFiles.sort(key=len) # sorts by descending length
	isingFiles.sort() # sorts normally by alphabetical order
	isingFiles.sort(key=len) # sorts by descending length

	# for dataFile in rudyFiles:
	# 	print '\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
	# 	print 'input file: ' + dataFile
	# 	beginTime = time.time()
	# 	runTests('./data/rudy/'+ dataFile, minGenerations, crossoverChoice, mutationChoice, numtrials)
	# 	endTime = time.time()
	# 	print 'total runtime for all trials - ' + str(endTime - beginTime)
	# 	print str(float(rudyFiles.index(dataFile)+1)/len(rudyFiles)) + ' percent complete for rudyFiles'
	# 	print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'

	for dataFile in isingFiles:
		print '\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
		print 'input file: ' + dataFile
		beginTime = time.time()
		runTests('./data/ising/'+ dataFile, minGenerations, crossoverChoice, mutationChoice, numtrials)
		endTime = time.time()
		print 'total runtime for all trials - ' + str(endTime - beginTime)
		print str(float(isingFiles.index(dataFile)+1)/len(isingFiles)) + ' percent complete for isingFiles'
		print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'

def runTests(inputPath, minGenerations, crossoverChoice, mutationChoice, numtrials):
	random.seed()

	cityGraph = initializeGraph(inputPath)
	# illustrateFullGraph(cityGraph)

	if(crossoverChoice == '1'):
		crossover = crossover1
	else:
		crossover = crossover2

	if(mutationChoice == '1'):
		mutation = mutation1
	else:
		mutation = mutation2

	geneticPerfTracker = []
	wocPerfTracker = []
	geneticTimeTracker = []
	wocTimeTracker = []

	for i in range(numtrials):
		print '------------'
		fitnessList = []
		print 'trial: ' + str(i + 1)
		pop = Population(cityGraph, 100, crossover, mutation, .1)
		start = time.time()
		fitnessList = runGeneticAlgorithm(cityGraph, pop, minGenerations)
		mid = time.time()
		woc = aggregate(cityGraph, pop)
		finish = time.time()

		geneticPerformance = pop.getFitness(pop.cuts[0])
		wocPerformance = pop.getFitness(woc)
		geneticRuntime = mid - start
		wocRuntime = finish - mid

		geneticPerfTracker.append(geneticPerformance)
		wocPerfTracker.append(wocPerformance)
		geneticTimeTracker.append(geneticRuntime)
		wocTimeTracker.append(wocRuntime)

		fitnessOverTime(fitnessList, inputPath, crossoverChoice, mutationChoice, i+1)
		print '------------'

	numNodes = cityGraph.number_of_nodes()
	numEdges = cityGraph.number_of_edges()
	geneticPerformanceAverage = getAvg(geneticPerfTracker)
	wocPerformanceAverage = getAvg(wocPerfTracker)
	geneticRuntimeAverage = getAvg(geneticTimeTracker)
	wocRuntimeAverage = getAvg(wocTimeTracker)
	print 'geneticPerformanceAverage: ' + str(geneticPerformanceAverage), '\twocPerformanceAverage: ' + str(wocPerformanceAverage)
	print 'geneticRuntimeAverage: ' + str(geneticRuntimeAverage), '\twocRuntimeAverage: ' + str(wocRuntimeAverage)

	writeData(inputPath, crossoverChoice, mutationChoice, geneticPerformanceAverage, geneticRuntimeAverage, wocPerformanceAverage, wocRuntimeAverage, numNodes, numEdges)

if __name__ == "__main__":
    main()
