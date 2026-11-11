# run the simulation data with the cost matrix M1. 
# M1 is the orignal cost matrix 
path_to_cost_matrix="cost_matrix/M1.csv"
outputdir="./output"
python run_WNMF.py -M ${path_to_cost_matrix} -k 10 --N 200 --outdir ${outputdir}