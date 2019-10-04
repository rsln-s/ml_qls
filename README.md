# MultiLevel Quantum Local Search

This repository contains the source code for the algorithm discussed in the paper "Multilevel Combinatorial Optimization Across Quantum Architectures".

### Installation

It is recommended to use Anaconda with intel channels. This requires intel compiler suite. Similar instructions should work with your preferred C++ compiler.

```
conda create --name ml_qls intelpython3_core python=3.6
conda activate ml_qls
conda install numpy scipy cython
conda install -c conda-forge nlopt
conda install scons
conda install -c etetoolkit argtable2
git clone https://github.com/rsln-s/ibmqxbackend.git 
cd ibmqxbackend
pip install -e .
cd ..
git clone https://github.com/rsln-s/ml_qls.git
cd ml_qls/qcommunity
pip install -e .
# Now, compile KaHIP
cd ../multilevel/coarsening/KaHIP
./compile
cd ../.. 
```

Now, edit `paths.sh` and set ML_QUANTUM_WORKING_DIR to be the path to ml_qls, e.g. `/home/rshaydu/quantum/ml_qls`

To test your installation, run the following example:

```
source path.sh
cd GP
python ml_gp.py ../data/GP/uk.graph
```


#### Common issues

```
./../coarsening/KaHiP/deploy/kaffpa_coarsen: error while loading shared libraries: libargtable2.so.0: cannot open shared object file: No such file or directory
```

Add path to Anaconda  lib to your LD_LIBRARY_PATH. For example:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rshaydu/soft/anaconda3/envs/ml_qls/lib   
```
