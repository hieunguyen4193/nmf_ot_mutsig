# run the simulation data with the cost matrix M1. 
# M1 is the orignal cost matrix 
path_to_cost_matrix="src/cost_matrix/M1.csv";
# path_to_cost_matrix="src/SBS_single_change_graph/graph_M1.csv";

costMatrixName=$(echo ${path_to_cost_matrix} | xargs -n 1 basename);

echo -e "working on ${costMatrixName}";
outputdir="./benchmark_all";
python src/run_WNMF.py -M ${path_to_cost_matrix} -k 10 --N 50 --outdir ${outputdir} --runName ${costMatrixName}
