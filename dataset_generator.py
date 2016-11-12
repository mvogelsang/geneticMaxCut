import math
import random 
import sys

# class to store x y info for node
class Node:
	def __init__(self, label, x, y): 
		self.label = label
		self.x = x
		self.y = y

# class to store 2 nodes	
class Edge: 
	def __init__(self, node1, node2):
		self.node1 = node1
		self.node2 = node2

# reads in files with node locations
def rd_file(tsp_file):
	node_list = [] 
	with open(tsp_file) as f:		
		for i in range(7):
			next(f)
		
		for line in f: 
			split = line.split(' ') 
			node_label = str(int(split[0]) - 1)
			node_x = float(split[1])
			node_y = float(split[2])
			
			new_node = Node(node_label, node_x, node_y)
			node_list.append(new_node)
		
		f.close()
		
	return node_list

# pull random combination from iterable
# from article at http://stackoverflow.com/questions/22229796/choose-at-random-from-combinations	
def random_combination(iterable, r): 
	pool = tuple(iterable)
	n = len(pool)
	indices = sorted(random.sample(xrange(n), r))
	return tuple(pool[i] for i in indices)

def check_edges(edges, new_edge):
	for edge in edges:	
		if(edge.node1.label == new_edge.node1.label and edge.node2.label == new_edge.node2.label):
					return False
		elif(edge.node1.label == new_edge.node2.label and edge.node2.label == new_edge.node1.label):
					return False
	
	return True
def generate_edges(node_list):
	edges = []
	
	# generate edge for each node
	for node in node_list:
		add = False
		new_edge = None
		
		while(add is False):
			edge_node = random.choice(node_list)
		
			if(edge_node.label != node.label):
				new_edge = Edge(node, edge_node)
				add = check_edges(edges, new_edge)
		
		edges.append(new_edge)
	
	num_edges = int(math.ceil(len(node_list) ** 1.25))
	
	while len(edges) != num_edges:
		new_edge = list(random_combination(node_list, 2))
		new_edge = Edge(new_edge[0], new_edge[1])
		
		add = check_edges(edges, new_edge)
		
		if(add is True):
			edges.append(new_edge)
	
	return edges

# write edge and node file
def write_file(node_list, edge_list, out_file):
	f = open(out_file, 'w')
	
	f.write('nodes:\n')
	
	for node in node_list:
		f.write(node.label + ' ' + str(node.x) + ' ' + str(node.y) + '\n')
	
	f.write('edges:\n')
	
	for edge in edge_list:
		f.write(edge.node1.label + ' ' + edge.node2.label + '\n')
	
	f.close()

def main():	
	random.seed() 
	
	in_file = sys.argv[1]
	out_file = sys.argv[2]  
	
	node_list = rd_file(in_file)
	edge_list = generate_edges(node_list)
	
	write_file(node_list, edge_list, out_file)

if __name__ == "__main__":
    main()



