##############################################################
#
#
# Genetic algorithm classes
#
#
##############################################################
class Population:
	def __init__(self, graph, population_size, crossover_func):
		self.graph = graph
		self.population_size = population_size
		self.crossover_func = crossover_func
	
	# generate random initial population
	def get_init_population(self):
	# put code shit
	
	# select parent for breeding
	def parent_select(self): 
	## put code shit
	
	# generate new population
	def breed_new_generation(self):
	## put code shit
	
	# sort population
	def sort_pop(self): 
	## put code shit
	
	# find fittest member of population
	def get_fittest(self): 
	## put code shit
	
	##def crossover(cut1, cut2): 
	## FILL IN LATER
	
class Cut:
	def __init__(self, cut, mutate_func):
		self.cut = cut
		self.fitness = self.get_fitness()
		self.mutate_func = mutate_func
	
	# get fitness for stored cut	
	def get_fitness(self): 
	## put code shit
	
	##def mutate(self):
	## FILL IN LATER
	
def 
