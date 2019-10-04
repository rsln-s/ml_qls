The PETSc installation is required for MLAMG. Please please refer to [LINK](https://www.mcs.anl.gov/petsc/documentation/installation.html)

Install:
Run the make command, then run the bash script run.sh. It will run all the required steps.
The 4 parameters which are required to run.sh are:

1- path to graph files: e.g. ./graphs/3elt.graph.levels/     
2- graph file name without extension e.g. 3elt
3- stop criteria for coarsening or max number of nodes in the coarsest level e.g. 10
4- interpolation order (max number of nodes which may participate in an aggregate) e.g. 1

You can pass more parameters to coarsening (mlamg). For furthur information, please refer to [MLSVM repo](https://github.com/esadr/mlsvm).
