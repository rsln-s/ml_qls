ALL: mlamg
CC 	 = g++ -L. 
CFLAGS 	 = -I.	
CPPFLAGS = -std=c++11 -g -O3     -w # -W -Wall -Weffc++ -Wextra -pedantic -O3
LOCDIR   = .
#MAIN 	 = mlsvm_classifier.cc
MAIN 	 = mlamg.cc
MANSEC   = Mat

LIBS= -lpugixml -lm -fopenmp 
LIBFLANN= /usr/local/lib/libflann_cpp_s.a

MLSVM_SRCS = pugixml.cc etimer.cc common_funcs.cc OptionParser.cc k_fold.cc svm_weighted.cc config_params.cc model_selection.cc solver.cc partitioning.cc refinement.cc  main_recursion.cc coarsening.cc loader.cc ds_node.cc ds_graph.cc mlsvm_classifier.cc
MLSVM_OBJS = $(MLSVM_SRCS:.cc=.o)

MLAMG_SRCS = pugixml.cc etimer.cc common_funcs.cc OptionParser.cc config_params.cc partitioning.cc main_recursion.cc coarsening.cc loader.cc ds_node.cc ds_graph.cc mlamg.cc
MLAMG_OBJS = $(MLAMG_SRCS:.cc=.o)

MlsvmSepTest_SRCS = pugixml.cc etimer.cc common_funcs.cc OptionParser.cc k_fold.cc svm_weighted.cc config_params.cc model_selection.cc solver.cc partitioning.cc refinement.cc  main_recursion.cc coarsening.cc loader.cc ds_node.cc ds_graph.cc mlsvmClassifierSepTest.cc
MlsvmSepTest_OBJS = $(MlsvmSepTest_SRCS:.cc=.o)

MlsvmNoTest_SRCS = pugixml.cc etimer.cc common_funcs.cc OptionParser.cc k_fold.cc svm_weighted.cc config_params.cc model_selection.cc solver.cc partitioning.cc refinement.cc  main_recursion.cc coarsening.cc loader.cc ds_node.cc ds_graph.cc mlsvmClassifierNoTest.cc
MlsvmNoTest_OBJS = $(MlsvmNoTest_SRCS:.cc=.o)



CSV_PETSC_SRCS= pugixml.cc config_params.cc etimer.cc common_funcs.cc OptionParser.cc convertor.cc loader.cc k_fold.cc  ./tools/mlsvm_csv_petsc.cc
CSV_PETSC_OBJS = $(CSV_PETSC_SRCS:.cc=.o)

KNN_SRCS= pugixml.cc config_params.cc etimer.cc common_funcs.cc OptionParser.cc loader.cc k_fold.cc  ./tools/mlsvm_knn.cc
KNN_OBJS = $(KNN_SRCS:.cc=.o) $(LIBFLANN) 

PERS_SRCS= pugixml.cc config_params.cc etimer.cc common_funcs.cc OptionParser.cc loader.cc k_fold.cc svm_unweighted.cc solver.cc model_selection.cc personalized.cc personalized_main.cc
PERS_OBJS = $(PERS_SRCS:.cc=.o)



include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


info: 
	-@echo $(LIBS)
	-@echo $(MLSVM_OBJS)
#pugixml: g++ -c pugixml.cpp pugixml.hpp
pugixml: mpicxx -c pugixml.cpp pugixml.hpp


mlsvm_classifier: $(MLSVM_OBJS) chkopts
	-${CLINKER} $(MLSVM_OBJS)  ${PETSC_MAT_LIB} -o mlsvm_classifier
	${RM} mlsvm_classifier.o 

mlamg: $(MLAMG_OBJS) chkopts
	-${CLINKER} $(MLAMG_OBJS)  ${PETSC_MAT_LIB} -o mlamg
	${RM} mlamg.o 
	
mlsvmSepTestClassifier: $(MlsvmSepTest_OBJS) chkopts
	-${CLINKER} $(MlsvmSepTest_OBJS)  ${PETSC_MAT_LIB} -o mlsvmSepTestClassifier
	${RM} mlsvmSepTestClassifier.o 

mlsvmNoTestClassifier: $(MlsvmNoTest_OBJS) chkopts
	-${CLINKER} $(MlsvmNoTest_OBJS)  ${PETSC_MAT_LIB} -o mlsvmNoTestClassifier
	${RM} mlsvmNoTestClassifier.o 
mlsvm_zscore: $(ZSCORE_OBJS) chkopts
	-${CLINKER} $(ZSCORE_OBJS)  ${PETSC_MAT_LIB} -o mlsvm_zscore
	${RM} mlsvm_zscore.o

mlsvm_csv_petsc: $(CSV_PETSC_OBJS) chkopts
	-${CLINKER} $(CSV_PETSC_OBJS)  ${PETSC_MAT_LIB} -o mlsvm_csv_petsc
	${RM} mlsvm_csv_petsc.o

mlsvm_knn: $(KNN_OBJS) chkopts
	-${CLINKER}  $(KNN_OBJS) ${PETSC_MAT_LIB} -o mlsvm_knn
	${RM} mlsvm_knn.o
	
pers: $(PERS_OBJS) chkopts
	-${CLINKER} $(PERS_OBJS)  ${PETSC_MAT_LIB} -o pers
	${RM} pers.o 

mlsvm_save_knn: $(mlsvm_Save_knn_OBJS) chkopts
	-${CLINKER} $(mlsvm_Save_knn_OBJS)  ${PETSC_MAT_LIB} -o mlsvm_save_knn
	${RM} mlsvm_save_knn.o 

clean_main:	#PETSc Makefile has clean method which conflict with clean method here, therefore this one is created
	make clean
	rm mlsvm_classifier
	echo "clean is done"
