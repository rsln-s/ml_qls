# Coarsening with KaHIP

## Step 1
Initial setup. Need to add static path to write coarsened graphs and maps


In file `multilevel_quantum/coarsening/KaHiP/lib/partition/coarsening/coarsening.cpp`
Edit line `117` and `128` accordingly.

In file `multilevel_quantum/coarsening/KaHiP/app`
Edit line `90` accordingly.

## Step 2
To install, run
```
cd KaHiP/
./compile.sh
```

## Test
To test, run
```
cd deploy
./kaffpa_coarsen ../examples/delaunay_n15.graph --k=2 --preconfiguration=fast
```
Check the directory `multilevel_quantum/coarsening/coarsened_graphs/graphs/` for coarsened graphs. If dir is empty, then step 1 needs to be redone.
