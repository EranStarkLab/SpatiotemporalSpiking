import numpy as np
import math
from collections import deque

from constants import NUM_CHANNELS, COORDINATES


def euclidean_dist(point_a, point_b):
    """
    inputs:
    pointA: (x,y) tuple representing a point in 2D space
    pointB: (x,y) tuple representing a point in 2D space

    returns:
    The euclidean distance between the points
    """
    return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)

def calculate_distances_matrix(coordinates):
    """
    inputs:
    coordinates: a list of (x, y) tuples representing the coordinates of different channels

    returns:
    A 2D matrix in which a cell (i, j) contains the distance from coordinate i to coordinate j
    """
    distances = np.zeros((NUM_CHANNELS, NUM_CHANNELS))

    for i in range(NUM_CHANNELS):
        for j in range(NUM_CHANNELS):
            distances[i, j] = euclidean_dist(coordinates[i], coordinates[j])

    return distances

class Graph(object):
    """
    An object used to represent a graph
    """

    def __init__(self, start_nodes, end_nodes, graph_matrix):
        self.start_nodes = start_nodes
        self.end_nodes = end_nodes
        self.graph_matrix = graph_matrix
        self.reversed = False
        self.num_edges, self.edges = self._get_all_edges()

    def flip_graph(self):
        """
        Flips the values of the weights in the graph, i.e Positive weights will become negative
        """
        self.edges = {v: [(e[0], e[1], e[2] * -1) for e in self.edges[v]] for v in self.edges}
        self.reversed = not self.reversed

    def average_weight(self):
        """
        returns:
        The average weight in the graph
        """
        total = sum(sum(e[2] for e in self.edges[v]) for v in self.edges)
        counter = self.num_edges

        if counter == 0:
            return 0

        return total / counter

    def _get_all_edges(self):
        """
        returns:
        a list of all edges in the graph. each edge is represented using a tupple:
            (from node, to node, weight of edge)
        """
        edges = dict()
        counter = 0
        # Check to see if the graph is currently reversed and determine what is the value that will
        # be used to indicate that an edge doesn't exist
        if self.reversed:
            not_exist = 1
        else:
            not_exist = -1

        for i in range(NUM_CHANNELS):
            for j in range(NUM_CHANNELS):
                if self.graph_matrix[i, j] != not_exist:
                    if i in edges:
                        edges[i].append((i, j, self.graph_matrix[i, j]))
                    else:
                        edges[i] = [(i, j, self.graph_matrix[i, j])]
                    counter += 1
        return counter, edges

    def _dag_shortest_path(self, src_node):
        """
        An implementation of the Bellman Ford algorithm to find shortest distances in a graph

        inputs:
        src_node: The src node from which the search will begin

        returns:
        A list of size |nodes| containing the minimal distance from the source node to each other node
        """
        size_v = NUM_CHANNELS  # this is |V|
        dists = [float('inf') for _ in range(size_v)]
        dists[src_node] = 0

        edges = self.edges
        q = deque()
        q += edges[src_node]
        visited = {src_node}

        while len(q) > 0:
            origin, dest, w = q.popleft()
            if dists[dest] > dists[origin] + w:
                dists[dest] = dists[origin] + w
            if dest not in visited:
                visited.add(dest)
                if dest in edges:
                    q += edges[dest]

        return dists

    def _find_minimum_dist_to_end_nodes(self, dists):
        """
        inputs:
        dists: the output of the Bellman Ford algorithm

        returns:
        The minimum distance to any of the end nodes of the graph
        """
        minimum = float('inf')
        for end_node in self.end_nodes:
            if dists[end_node] < minimum:
                minimum = dists[end_node]
        return minimum

    def shortest_distance_from_src_to_end(self):
        """
        returns:
        The shortest distance from any source node to any end node
        """
        total_min_dist = float('inf')
        for src_node in self.start_nodes:
            short_dists = self._dag_shortest_path(src_node)
            min_dist = self._find_minimum_dist_to_end_nodes(short_dists)
            if min_dist < total_min_dist:
                total_min_dist = min_dist
        return total_min_dist

    def longest_distance_from_src_to_end(self):
        """
        returns:
        The longest distance from any source node to any end node
        """
        self.flip_graph()
        longest = self.shortest_distance_from_src_to_end()
        self.flip_graph()
        return longest * -1


class DepolarizationGraph(object):
    """
    This feature tries to estimate the way the signal traverses between the different channels. This traversal is
    modeled into a graph, where each node indicates a channel in a certain time, and each edge represents the speed in
    which the signal travels between the two channels that comprise it.
    """

    def __init__(self, thr=0.25, data_name='dep'):
        self.thr = thr

        self.name = 'Graph'
        self.data_name = data_name

    def set_data(self, new_data):
        self.data_name = new_data

    def calculate_feature(self, spike_lst, amps):
        """
        inputs:
        spike_lst: A list of Spike object that the feature will be calculated upon.
        amps: Relative amplitudes corresponding to the spike_lst parameter, used to filter out channels

        returns:
        A matrix in which entry (i, j) refers to the j metric of Spike number i.
        """
        # Determine the (x,y) coordinates of the NUM_CHANNELS different channels and calculate the distances matrix
        coordinates = COORDINATES
        dists = calculate_distances_matrix(coordinates)
        result = np.zeros((len(spike_lst), 3))

        for index, spike_amp in enumerate(zip(spike_lst, amps)):
            spike, amp = spike_amp
            arr = spike.data
            threshold = self.thr * amp.max()  # Setting the threshold to be self.thr the size of max depolarization

            g_temp = []
            for i in range(NUM_CHANNELS):
                max_dep_index = arr[i].argmin()
                if amp[i] >= threshold:
                    g_temp.append((i, max_dep_index))
            g_temp.sort(key=lambda x: x[1])
            assert len(g_temp) > 0

            graph_matrix = np.ones((NUM_CHANNELS, NUM_CHANNELS)) * (-1)
            for i, (channel1, timestep1) in enumerate(g_temp):
                for channel2, timestep2 in g_temp[i + 1:]:
                    if timestep2 != timestep1:
                        velocity = dists[channel1, channel2] / (timestep2 - timestep1)
                        graph_matrix[channel1][channel2] = velocity

            # The first nodes that reached depolarization
            start_nodes = [channel for (channel, timestep) in g_temp if timestep == g_temp[0][1]]
            # The last nodes that reached depolarization
            end_nodes = [channel for (channel, timestep) in g_temp if timestep == g_temp[-1][1]]
            graph = Graph(start_nodes, end_nodes, graph_matrix)
            if graph.num_edges == 0:
                continue

            # Calculate features from the graph
            result[index, 0] = graph.average_weight()
            result[index, 1] = graph.shortest_distance_from_src_to_end()
            result[index, 2] = graph.longest_distance_from_src_to_end()

        return result

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return [f"{self.data_name}_{self.name}_Average_weight", f"{self.data_name}_{self.name}_Shortest_path",
                f"{self.data_name}_{self.name}_Longest_path"]
