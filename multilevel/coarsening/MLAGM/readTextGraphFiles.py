'''
Author: Ehsan Sadrfaridpour
Date: Dec 8, 2018
Purpose: create a weighted adjacency matrix in PETSc binary format

input: graph in text format

output: WA matrix
'''


import sys
import numpy as np
import scipy.sparse as ss
import argparse
import pdb

from os import environ
dirs = environ['PETSC_DIR']
sys.path.insert(0, dirs+'/bin/')
import PetscBinaryIO
io = PetscBinaryIO.PetscBinaryIO()

class TextGraph2Mat:
    def __init__(self, data_path, filename):
        self.graph_path = data_path + '/'
        self.graph_file = filename

    def loadGraph(self):
        all_lines = []
        with open(self.graph_path + self.graph_file, 'r') as in_file:
            all_lines = in_file.readlines()
        # remove the '\n' from the end of each line
        for idx in range(len(all_lines)):
            all_lines[idx] = all_lines[idx].replace('\n','')

        self.num_nodes = int(all_lines[0].split(' ')[0])
        self.num_edges = int(all_lines[0].split(' ')[1])

        nodes = {}
        for idx, line in enumerate(all_lines[1:]):
            for nei in line.split(' '):
                if nei == '': continue
                node_id = idx + 1
                nei_id = int(nei)
                if node_id in nodes:
                    nodes[node_id].add(nei_id)
                else:
                    nodes[node_id] = {nei_id}
        self.graph_adj_dict = nodes

    def calcAdjacencyMatrix(self):
        nd_A = np.zeros((self.num_nodes, self.num_nodes), dtype=int)

        for node, neigh in self.graph_adj_dict.items():
            for nei in neigh:
                nd_A[node-1,nei-1] = 1

        num_calc_edge = np.sum(nd_A) // 2
        # print(f'number of calculated edges:{num_calc_edge}')
        assert(num_calc_edge == self.num_edges),\
            "number of edges doesn't match"

        self.mat_A = ss.csr_matrix(nd_A)


    def writeMat2PetscFormat(self):
        graph_identifier = self.graph_file.split('.')[0]
        output_filename = f'{self.graph_path}/input_WA_{graph_identifier}.dat'
        io.writeBinaryFile(output_filename, [self.mat_A,])
        print(f'Export {output_filename} finished successfully')

    def main(self):
        # load the graph
        self.loadGraph()
        self.calcAdjacencyMatrix()
        self.writeMat2PetscFormat()
        # pdb.set_trace()
        print('Finished successfully')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path",
                        action="store",
                        dest="data_path",
                        help="Path to data files")
    parser.add_argument("-f", "--filename",
                        action="store",
                        dest="filename",
                        help="graph file name")

    args = parser.parse_args()
    assert(args.data_path), \
        "Path to data files is required! -d"
    assert(args.filename), "file name is required! -f"

    tgm = TextGraph2Mat(args.data_path, args.filename)
    tgm.main()

if __name__ == "__main__":
    main()
