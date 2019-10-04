# Stores the state of the multilevel solver and dumps it to the disk in the end
# Not a singleton anymore, so be careful!

import pickle
from collections import namedtuple
from operator import attrgetter

# Solver call index is number from 0 to total_num_solver_calls indicating to which solver call this record corresponds
RefinementStepRecord = namedtuple('RefinementStepRecord', ['level', 'obj', 'imbalance01', 'imbalance10', 'energy', 'solver_call_index', 'solution_before', 'free_nodes', 'solution_after'])

class StateWatcher:
    def __init__(self):
        self.current_level = None
        self.num_solver_calls = 0
        self.args = None
        self.var_forms = {}
        self.__all_levels = []
    
    # various stuff needed for restart from checkpoint
        self.all_graphs = None
        self.up_maps = None
        self.lowest_level = None
        self.graph_file = None
        self.ptn_variables = None
        self.outpath = None
        self.coarsest_level = None
    
    # imbalance01 = part0/part1, imbalance10,part1/part0 
    def record_refinement_step(self, obj, imbalance01, imbalance10, energy, solution_before, free_nodes, solution_after):
        print("Recording: ", energy, imbalance01, imbalance10,obj, self.num_solver_calls, "at level", self.current_level)
        self.__all_levels.append(RefinementStepRecord(level=self.current_level, obj=obj, imbalance01=imbalance01, imbalance10=imbalance10, energy=energy, solver_call_index=self.num_solver_calls, solution_before=solution_before, solution_after=solution_after, free_nodes=free_nodes))
    
    def print_records(self):
        print("Total solver calls: ", self.num_solver_calls)
        for r in sorted(self.__all_levels, key=attrgetter('solver_call_index')):
            print("level {} \t obj {} \t imbal01 {:4f} \t imbal10 {:4f} ".format(r.level, r.obj, r.imbalance10, r.imbalance01))
    
    def save_final_res_to_disk(self, fpath):
        print("Saving results to {}".format(fpath))
        pickle.dump((self.__all_levels, self.num_solver_calls, self.args), open(fpath, "wb"))

    def save_final_res_to_disk_hayato(self, fpath):
        expfile = open("myexperiment.txt", "a")
        #graph  solver  hardware-size   problem-type alpha beta refine-method   seed    imbal01 imbal10    objective-value number_solver_calls
        out = [
                self.graphname,
                self.args.solver,
                str(self.args.hardware_size),
                self.args.problem_type,
                str(self.args.alpha),
                str(self.args.beta),
                self.args.refine_method,
                str(self.args.seed),
                str(round(self.imbalance10, 5)),
                str(round(self.imbalance01, 5)),
                str(round(self.obj_value, 7)),
                str(self.num_solver_calls)
                
            ]
        out = "\t".join(out) + "\n"
        expfile.write(out)
        expfile.close()
