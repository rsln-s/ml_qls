'''
Author: Ehsan Sadrfaridpour
Date: Dec 8, 2018
Purpose: create a weighted adjacency matrix in PETSc binary format

input: Adjacency matrix, P matrix, volume vector in PETSc format
output: graph in text format, mapping, strenght


PETSc matrix detail:
class MatSparse(tuple):
    """Mat represented as CSR tuple ((M, N), (rowindices, col, val))
    This should be instantiated from a tuple:
    mat = MatSparse( ((M,N), (rowindices,col,val)) )


Output format desc:
row 1: # nodes, # edges, 11 (don't care about this number)
rows 2 ... end: each row corresponds to a node and its neighbors

first letter should be either 'c' (for coarse seed) or 'f' (for a non-seed node)

if it is 'c' then the next number will be an id of this aggregate at the coarse level; if it is 'f' then no coarse id is given

the next number is a volume of this node

then there is a list of neighbors in a format neighbor_id and edge weight

rows 2...end are sorted by node id

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

class Mat2Text:
    def __init__(self, data_path, level):
        self.data_path = data_path + '/'
        self.level = level

    def createCsrMatrix(self, s_mat):
        assert(type(s_mat) == PetscBinaryIO.MatSparse), \
            "this function suppose to work only for PETSc matrix"
        return ss.csr_matrix((s_mat[1][2],s_mat[1][1], s_mat[1][0]),
                             shape=s_mat[0])

    def createDenseMatrix(self, s_mat):
        assert(type(s_mat) == PetscBinaryIO.MatSparse), \
            "this function suppose to work only for PETSc matrix"
        csr_mat = ss.csr_matrix((s_mat[1][2],s_mat[1][1], s_mat[1][0]),
                             shape=s_mat[0])
        return csr_mat.todense(), csr_mat.shape, csr_mat.nnz

    # load all the data
    def loadData(self):
        mat_WA = io.readBinaryFile(
            f'{self.data_path}/mat_l_{self.level}_WA.dat')[0]
        mat_P = io.readBinaryFile(
            f'{self.data_path}/mat_l_{self.level}_P.dat')[0]
        Vol = io.readBinaryFile(
            f'{self.data_path}/mat_l_{self.level}_vol.dat')[0]
        assert(type(Vol) == PetscBinaryIO.Vec), "wrong type"
        self.v_vol = np.asarray(Vol)

        self.dm_WA, self.dm_WA_shape, self.nnz = self.createDenseMatrix(mat_WA)
        self.dm_P, dm_P_shape, _ = self.createDenseMatrix(mat_P)

    # get the list of neighbors idx and edge weights
    def getNeighbors(self, idx):
        res = ''
        _, col_idx = np.nonzero(self.dm_WA[idx])
        for cidx in col_idx:
            res += f' {cidx} {self.dm_WA[idx, cidx]}'
        return res

    def calcMat2GraphCF(self):
        res = []
        for r in range(self.dm_WA_shape[0]):
            vol = self.v_vol[r]
            if r in self.dict_seed2fine: # seed point
                cidx = self.dict_seed2fine[r]
                res.append(f'c {cidx} {vol}{self.getNeighbors(r)}')
            else:               # fine point
                res.append(f'f {vol}{self.getNeighbors(r)}')
        return res



    def calcStrengthMapping(self):
        res = []
        for f , seed_info in self.dict_f2s_strength.items():
            tmp = f'{f}'
            for k in sorted(list(seed_info)):
                tmp += f' {k[0]} {k[1]}'
            res.append(tmp)
        return res

    def exportList(self, lst_content,  file_name):
        with open(self.data_path + file_name, 'w') as out:
            for line in lst_content:
                out.write(f'{line}\n')
        print(f'export {self.data_path + file_name} finished')

    def main(self):
        # load the data
        self.loadData()

        # list the seed nodes
        print(f'num nodes:{self.dm_WA_shape[0]}, num edges:{self.nnz//2}')
        self.dict_seed2fine = {}
        self.dict_f2s_strength = {}

        row_idx, col_idx = np.nonzero(self.dm_P)
        cord = zip(row_idx, col_idx)
        # r is fine node idx
        for r, c in cord:
            strength = (c, self.dm_P[r,c])
            if r in self.dict_f2s_strength:
                self.dict_f2s_strength[r].add(strength)
            else:
                self.dict_f2s_strength[r] = {strength}

        seed_elems = np.where(self.dm_P == 1)
        for f, s in zip(seed_elems[0], seed_elems[1]):
            self.dict_seed2fine[s] = f




        graph_seed_fine = self.calcMat2GraphCF()
        mapping_ratio = self.calcStrengthMapping()

        self.exportList(graph_seed_fine, f'level{self.level}.dat')
        self.exportList(mapping_ratio, f'map{self.level}.dat')
        # pdb.set_trace()
#        print('Finished successfully')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path",
                        action="store",
                        dest="data_path",
                        help="Path to data files")
    parser.add_argument("-l", "--level",
                        action="store",
                        dest="level",
                        help="level number")

    args = parser.parse_args()
    assert(args.data_path), \
        "Path to data files is required! -d"
    assert(args.level), "level number is required! -l"

    mt = Mat2Text(args.data_path, args.level)
    mt.main()

if __name__ == "__main__":
    main()
