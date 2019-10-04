/*
Author: Ehsan Sadrfaridpour
Date: Dec 8, 2018
*/

#include "etimer.h"
#include "loader.h"          //load the flann data into the WA matrix
#include "main_recursion.h"
#include "coarsening.h"         //coarse the WA matrix
#include "config_params.h"
#include "common_funcs.h"
#include <cassert>
#include "config_logs.h"

Config_params* Config_params::instance = NULL;

int main(int argc, char **argv)
{
    PetscInitialize(NULL, NULL, NULL, NULL);

    // read parameters
    paramsInst->read_params("./params.xml",
                              argc, argv,
                            paramsInst->amg_coarse_quantum);

    ETimer t_all;
    Mat m_P, m_WA;
    Vec v_vol;
    // load adjacency matrix file
    Loader ld;
    std::string mat_adj_fname = paramsInst->get_ds_path() +"/input_WA_" +
                                    paramsInst->get_ds_name()+ ".dat";
    std::cout << "input adjacency matrix file:" << mat_adj_fname << std::endl;

    m_WA = ld.load_matrix(mat_adj_fname);
    PetscInt num_row=0;
    MatGetSize(m_WA,&num_row,NULL);
    v_vol = ld.init_volume(1, num_row);

    // ================= Multilevel Solver ================
    ETimer t_solver;

    MainRecursion main_hierarchy;
    main_hierarchy.community_detection(m_P, m_WA, v_vol, 0);

    printf("[MLAMG] coarsening finished successfully\n");
    PetscFinalize();
    return 0;
}
