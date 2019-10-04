#!/bin/bash

function retry {
  local n=1
  local max=5
  local delay=10
  while true; do
    "$@" && break || {
      if [[ $n -lt $max ]]; then
        ((n++))
        echo "Command failed. Attempt $n/$max:"
        sleep $delay;
      else
        fail "The command has failed after $n attempts."
      fi
    }
  done
}

g=get_erdos_renyi_graph
l=10
r=10
ansatz_depth=2
label=test_setup
run=2
seed=4
graph_generator_seed=6
remove_edge=9
#problem="maxcut --weighted --zero-tol"
#problem="maxcut --weighted"
problem="maxcut --restart-local"
#problem="modularity"
#problem="maxcut --weighted --restart-local"
niter=50
#retry python -m qcommunity.optimization.optimize -g $g -l $l -r $r --method MLSL_NLOPT --localopt-method LN_BOBYQA --seed $seed --graph-generator-seed $graph_generator_seed --backend IBMQX --ansatz-depth $ansatz_depth --ansatz QAOA --label $label --max-active-runs $run --niter $niter --verbose --problem $problem 
#retry mpirun -np 2 python -m mpi4py -m qcommunity.optimization.optimize -g $g -l $l -r $r --method libensemble --localopt-method LN_BOBYQA --mpi --seed $seed --graph-generator-seed $graph_generator_seed --backend IBMQX --ansatz-depth $ansatz_depth --ansatz QAOA --label $label --max-active-runs $run --niter $niter --verbose --problem $problem
#retry python -m qcommunity.optimization.optimize -g $g -l $l -r $r --method COBYLA_NLOPT --seed $seed --graph-generator-seed $graph_generator_seed --backend IBMQX --ansatz-depth $ansatz_depth --ansatz QAOA --label $label --niter $niter --verbose --problem $problem
#python -m qcommunity.optimization.optimize -g $g -l $l -r $r --method BOBYQA_NLOPT --seed $seed --graph-generator-seed $graph_generator_seed --backend IBMQX --ansatz-depth $ansatz_depth --ansatz QAOA --label $label --niter $niter --verbose --problem $problem
#python -m qcommunity.optimization.optimize -g $g -l $l -r $r --method BOBYQA_NLOPT --seed $seed --graph-generator-seed $graph_generator_seed --backend IBMQX --ansatz-depth $ansatz_depth --ansatz QAOA --label $label --niter $niter --verbose --problem $problem --sample-points /zfs/safrolab/users/rshaydu/quantum/data/for_jeff/optimal_points/0415_get_connected_caveman_graph_l_2_r_6_seed_1_d2x2.pkl --remove-edge
#python -m qcommunity.optimization.optimize -g $g -l $l -r $r --method SAMI --seed $seed --graph-generator-seed $graph_generator_seed --backend IBMQX --ansatz-depth $ansatz_depth --ansatz QAOA --label $label --niter $niter --verbose --problem $problem
python optimize.py -g $g -l $l -r $r -p 0.4 --method BOBYQA_NLOPT --seed $seed --graph-generator-seed $graph_generator_seed --backend IBMQX --ansatz-depth $ansatz_depth --ansatz QAOA --label $label --niter $niter --niter-local 10 --verbose --problem $problem
#python -m qcommunity.optimization.optimize -g $g -l $l -r $r --method RANDOM --seed $seed --graph-generator-seed $graph_generator_seed --backend IBMQX --ansatz-depth $ansatz_depth --ansatz QAOA --label $label --niter $niter --verbose --problem $problem
#retry python -m qcommunity.optimization.optimize -g $g -l $l -r $r --method NELDERMEAD_NLOPT --seed $seed --graph-generator-seed $graph_generator_seed --backend IBMQX --ansatz-depth $ansatz_depth --ansatz QAOA --label $label --niter $niter --verbose --problem $problem
#retry python -m qcommunity.optimization.optimize -g $g -l $l -r $r --method NEWUOA_NLOPT --seed $seed --graph-generator-seed $graph_generator_seed --backend IBMQX --ansatz-depth $ansatz_depth --ansatz QAOA --label $label --niter $niter --verbose --problem $problem
#retry python -m qcommunity.optimization.optimize -g $g -l $l -r $r --method PRAXIS_NLOPT --seed $seed --graph-generator-seed $graph_generator_seed --backend IBMQX --ansatz-depth $ansatz_depth --ansatz QAOA --label $label --niter $niter --verbose --problem $problem
#retry python -m qcommunity.optimization.optimize -g $g -l $l -r $r --method SBPLX_NLOPT --seed $seed --graph-generator-seed $graph_generator_seed --backend IBMQX --ansatz-depth $ansatz_depth --ansatz QAOA --label $label --niter $niter --verbose --problem $problem
