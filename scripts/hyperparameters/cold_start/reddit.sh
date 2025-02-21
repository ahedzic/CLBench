cd benchmarking/cold_start

# Leroy
python main_cold_leroy.py  --data_name reddit  --model Leroy --epochs 1 --kill_cnt 1 --eval_steps 1  --runs 1 --cold_perc 0.0 > output_reddit_leroy_true

## ACCSLP
python main_cold_accslp.py  --data_name reddit  --model ACCSLP  --alpha 20 --beta 27 --rank 5 --groups 41 --max_nodes 247 --epochs 1 --kill_cnt 1 --eval_steps 1  --runs 1 --cold_perc 0.0 > output_reddit_accslp_true
python main_cold_accslp.py  --data_name reddit  --model ACCSLP  --alpha 20 --beta 27 --rank 5 --groups 41 --max_nodes 247 --epochs 1 --kill_cnt 1 --eval_steps 1  --runs 1 --cold_perc 0.25 > output_reddit_accslp_25_edge
python main_cold_accslp.py  --data_name reddit  --model ACCSLP  --alpha 20 --beta 27 --rank 5 --groups 41 --max_nodes 247 --epochs 1 --kill_cnt 1 --eval_steps 1  --runs 1 --cold_perc 0.25 --blind node > output_reddit_accslp_25_node
python main_cold_accslp.py  --data_name reddit  --model ACCSLP  --alpha 20 --beta 27 --rank 5 --groups 41 --max_nodes 247 --epochs 1 --kill_cnt 1 --eval_steps 1  --runs 1 --cold_perc 0.50 > output_reddit_accslp_50_edge
python main_cold_accslp.py  --data_name reddit  --model ACCSLP  --alpha 20 --beta 27 --rank 5 --groups 41 --max_nodes 247 --epochs 1 --kill_cnt 1 --eval_steps 1  --runs 1 --cold_perc 0.50 --blind node > output_reddit_accslp_50_node
python main_cold_accslp.py  --data_name reddit  --model ACCSLP  --alpha 20 --beta 27 --rank 5 --groups 41 --max_nodes 247 --epochs 1 --kill_cnt 1 --eval_steps 1  --runs 1 --cold_perc 0.75 > output_reddit_accslp_75_edge
python main_cold_accslp.py  --data_name reddit  --model ACCSLP  --alpha 20 --beta 27 --rank 5 --groups 41 --max_nodes 247 --epochs 1 --kill_cnt 1 --eval_steps 1  --runs 1 --cold_perc 0.75 --blind node > output_reddit_accslp_75_node
python main_cold_accslp.py  --data_name reddit  --model ACCSLP  --alpha 20 --beta 27 --rank 5 --groups 41 --max_nodes 247 --epochs 1 --kill_cnt 1 --eval_steps 1  --runs 1 --cold_perc 0.90 > output_reddit_accslp_90_edge
python main_cold_accslp.py  --data_name reddit  --model ACCSLP  --alpha 20 --beta 27 --rank 5 --groups 41 --max_nodes 247 --epochs 1 --kill_cnt 1 --eval_steps 1  --runs 1 --cold_perc 0.90 --blind node > output_reddit_accslp_90_node

## CN
python main_cold_heuristic.py --data_name reddit --use_heuristic CN --cold_perc 0.0 > output_reddit_cn_true
python main_cold_heuristic.py --data_name reddit --use_heuristic CN --cold_perc 0.25 > output_reddit_cn_25_edge
python main_cold_heuristic.py --data_name reddit --use_heuristic CN --cold_perc 0.25 --blind node > output_reddit_cn_25_node
python main_cold_heuristic.py --data_name reddit --use_heuristic CN --cold_perc 0.50 > output_reddit_cn_50_edge
python main_cold_heuristic.py --data_name reddit --use_heuristic CN --cold_perc 0.50 --blind node > output_reddit_cn_50_node
python main_cold_heuristic.py --data_name reddit --use_heuristic CN --cold_perc 0.75 > output_reddit_cn_75_edge
python main_cold_heuristic.py --data_name reddit --use_heuristic CN --cold_perc 0.75 --blind node > output_reddit_cn_75_node
python main_cold_heuristic.py --data_name reddit --use_heuristic CN --cold_perc 0.90 > output_reddit_cn_90_edge
python main_cold_heuristic.py --data_name reddit --use_heuristic CN --cold_perc 0.90 --blind node > output_reddit_cn_90_node

## AA
python main_cold_heuristic.py --data_name reddit --use_heuristic AA --cold_perc 0.0 > output_reddit_aa_true
python main_cold_heuristic.py --data_name reddit --use_heuristic AA --cold_perc 0.25 > output_reddit_aa_25_edge
python main_cold_heuristic.py --data_name reddit --use_heuristic AA --cold_perc 0.25 --blind node > output_reddit_aa_25_node
python main_cold_heuristic.py --data_name reddit --use_heuristic AA --cold_perc 0.50 > output_reddit_aa_50_edge
python main_cold_heuristic.py --data_name reddit --use_heuristic AA --cold_perc 0.50 --blind node > output_reddit_aa_50_node
python main_cold_heuristic.py --data_name reddit --use_heuristic AA --cold_perc 0.75 > output_reddit_aa_75_edge
python main_cold_heuristic.py --data_name reddit --use_heuristic AA --cold_perc 0.75 --blind node > output_reddit_aa_75_node
python main_cold_heuristic.py --data_name reddit --use_heuristic AA --cold_perc 0.90 > output_reddit_aa_90_edge
python main_cold_heuristic.py --data_name reddit --use_heuristic AA --cold_perc 0.90 --blind node > output_reddit_aa_90_node

## RA
python main_cold_heuristic.py --data_name reddit --use_heuristic RA --cold_perc 0.0 > output_reddit_ra_true
python main_cold_heuristic.py --data_name reddit --use_heuristic RA --cold_perc 0.25 > output_reddit_ra_25_edge
python main_cold_heuristic.py --data_name reddit --use_heuristic RA --cold_perc 0.25 --blind node > output_reddit_ra_25_node
python main_cold_heuristic.py --data_name reddit --use_heuristic RA --cold_perc 0.50 > output_reddit_ra_50_edge
python main_cold_heuristic.py --data_name reddit --use_heuristic RA --cold_perc 0.50 --blind node > output_reddit_ra_50_node
python main_cold_heuristic.py --data_name reddit --use_heuristic RA --cold_perc 0.75 > output_reddit_ra_75_edge
python main_cold_heuristic.py --data_name reddit --use_heuristic RA --cold_perc 0.75 --blind node > output_reddit_ra_75_node
python main_cold_heuristic.py --data_name reddit --use_heuristic RA --cold_perc 0.90 > output_reddit_ra_90_edge
python main_cold_heuristic.py --data_name reddit --use_heuristic RA --cold_perc 0.90 --blind node > output_reddit_ra_90_node

## shortest path
python main_cold_heuristic.py --data_name reddit --use_heuristic shortest_path --cold_perc 0.0 > output_reddit_shortest_true
python main_cold_heuristic.py --data_name reddit --use_heuristic shortest_path --cold_perc 0.25 > output_reddit_shortest_25_edge
python main_cold_heuristic.py --data_name reddit --use_heuristic shortest_path --cold_perc 0.25 --blind node > output_reddit_shortest_25_node
python main_cold_heuristic.py --data_name reddit --use_heuristic shortest_path --cold_perc 0.50 > output_reddit_shortest_50_edge
python main_cold_heuristic.py --data_name reddit --use_heuristic shortest_path --cold_perc 0.50 --blind node > output_reddit_shortest_50_node
python main_cold_heuristic.py --data_name reddit --use_heuristic shortest_path --cold_perc 0.75 > output_reddit_shortest_75_edge
python main_cold_heuristic.py --data_name reddit --use_heuristic shortest_path --cold_perc 0.75 --blind node > output_reddit_shortest_75_node
python main_cold_heuristic.py --data_name reddit --use_heuristic shortest_path --cold_perc 0.90 > output_reddit_shortest_90_edge
python main_cold_heuristic.py --data_name reddit --use_heuristic shortest_path --cold_perc 0.90 --blind node > output_reddit_shortest_90_node

## katz
python main_cold_heuristic.py --data_name reddit --use_heuristic katz_close --cold_perc 0.0 > output_reddit_katz_true
python main_cold_heuristic.py --data_name reddit --use_heuristic katz_close --cold_perc 0.25 > output_reddit_katz_25_edge
python main_cold_heuristic.py --data_name reddit --use_heuristic katz_close --cold_perc 0.25 --blind node > output_reddit_katz_25_node
python main_cold_heuristic.py --data_name reddit --use_heuristic katz_close --cold_perc 0.50 > output_reddit_katz_50_edge
python main_cold_heuristic.py --data_name reddit --use_heuristic katz_close --cold_perc 0.50 --blind node > output_reddit_katz_50_node
python main_cold_heuristic.py --data_name reddit --use_heuristic katz_close --cold_perc 0.75 > output_reddit_katz_75_edge
python main_cold_heuristic.py --data_name reddit --use_heuristic katz_close --cold_perc 0.75 --blind node > output_reddit_katz_75_node
python main_cold_heuristic.py --data_name reddit --use_heuristic katz_close --cold_perc 0.90 > output_reddit_katz_90_edge
python main_cold_heuristic.py --data_name reddit --use_heuristic katz_close --cold_perc 0.90 --blind node > output_reddit_katz_90_node

#GCN
python main_cold_gnn.py  --data_name reddit  --input_size 602 --gnn_model GCN  --lr 0.001 --dropout 0.5 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.0 > output_reddit_gcn_true
python main_cold_gnn.py  --data_name reddit  --input_size 602 --gnn_model GCN  --lr 0.001 --dropout 0.5 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 > output_reddit_gcn_25_edge
python main_cold_gnn.py  --data_name reddit  --input_size 602 --gnn_model GCN  --lr 0.001 --dropout 0.5 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 --blind node > output_reddit_gcn_25_node
python main_cold_gnn.py  --data_name reddit  --input_size 602 --gnn_model GCN  --lr 0.001 --dropout 0.5 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.50 > output_reddit_gcn_50_edge
python main_cold_gnn.py  --data_name reddit  --input_size 602 --gnn_model GCN  --lr 0.001 --dropout 0.5 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.50 --blind node > output_reddit_gcn_50_node
python main_cold_gnn.py  --data_name reddit  --input_size 602 --gnn_model GCN  --lr 0.001 --dropout 0.5 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 > output_reddit_gcn_75_edge
python main_cold_gnn.py  --data_name reddit  --input_size 602 --gnn_model GCN  --lr 0.001 --dropout 0.5 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 --blind node > output_reddit_gcn_75_node
python main_cold_gnn.py  --data_name reddit  --input_size 602 --gnn_model GCN  --lr 0.001 --dropout 0.5 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.90 > output_reddit_gcn_90_edge
python main_cold_gnn.py  --data_name reddit  --input_size 602 --gnn_model GCN  --lr 0.001 --dropout 0.5 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.90 --blind node > output_reddit_gcn_90_node

#RGCN
python main_cold_relational_gnn.py  --data_name reddit  --input_size 602 --gnn_model RGCN  --lr 0.001 --dropout 0.5 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.0 > output_reddit_rgcn_true
python main_cold_relational_gnn.py  --data_name reddit  --input_size 602 --gnn_model RGCN  --lr 0.001 --dropout 0.5 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 > output_reddit_rgcn_25_edge
python main_cold_relational_gnn.py  --data_name reddit  --input_size 602 --gnn_model RGCN  --lr 0.001 --dropout 0.5 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 --blind node > output_reddit_rgcn_25_node
python main_cold_relational_gnn.py  --data_name reddit  --input_size 602 --gnn_model RGCN  --lr 0.001 --dropout 0.5 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.50 > output_reddit_rgcn_50_edge
python main_cold_relational_gnn.py  --data_name reddit  --input_size 602 --gnn_model RGCN  --lr 0.001 --dropout 0.5 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.50 --blind node > output_reddit_rgcn_50_node
python main_cold_relational_gnn.py  --data_name reddit  --input_size 602 --gnn_model RGCN  --lr 0.001 --dropout 0.5 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 > output_reddit_rgcn_75_edge
python main_cold_relational_gnn.py  --data_name reddit  --input_size 602 --gnn_model RGCN  --lr 0.001 --dropout 0.5 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 --blind node > output_reddit_rgcn_75_node
python main_cold_relational_gnn.py  --data_name reddit  --input_size 602 --gnn_model RGCN  --lr 0.001 --dropout 0.5 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.90 > output_reddit_rgcn_90_edge
python main_cold_relational_gnn.py  --data_name reddit  --input_size 602 --gnn_model RGCN  --lr 0.001 --dropout 0.5 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.90 --blind node > output_reddit_rgcn_90_node

#GAT
python main_cold_gnn.py  --data_name reddit  --input_size 602 --gnn_model GAT  --lr 0.001 --dropout 0.1 --l2 1e-4 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.0 > output_reddit_gat_true
python main_cold_gnn.py  --data_name reddit  --input_size 602 --gnn_model GAT  --lr 0.001 --dropout 0.1 --l2 1e-4 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 > output_reddit_gat_25_edge
python main_cold_gnn.py  --data_name reddit  --input_size 602 --gnn_model GAT  --lr 0.001 --dropout 0.1 --l2 1e-4 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 --blind node > output_reddit_gat_25_node
python main_cold_gnn.py  --data_name reddit  --input_size 602 --gnn_model GAT  --lr 0.001 --dropout 0.1 --l2 1e-4 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.50 > output_reddit_gat_50_edge
python main_cold_gnn.py  --data_name reddit  --input_size 602 --gnn_model GAT  --lr 0.001 --dropout 0.1 --l2 1e-4 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.50 --blind node > output_reddit_gat_50_node
python main_cold_gnn.py  --data_name reddit  --input_size 602 --gnn_model GAT  --lr 0.001 --dropout 0.1 --l2 1e-4 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 > output_reddit_gat_75_edge
python main_cold_gnn.py  --data_name reddit  --input_size 602 --gnn_model GAT  --lr 0.001 --dropout 0.1 --l2 1e-4 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 --blind node > output_reddit_gat_75_node
python main_cold_gnn.py  --data_name reddit  --input_size 602 --gnn_model GAT  --lr 0.001 --dropout 0.1 --l2 1e-4 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.90 > output_reddit_gat_90_edge
python main_cold_gnn.py  --data_name reddit  --input_size 602 --gnn_model GAT  --lr 0.001 --dropout 0.1 --l2 1e-4 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.90 --blind node > output_reddit_gat_90_node

#RGAT
python main_cold_relational_gnn.py  --data_name reddit  --input_size 602 --gnn_model RGAT  --lr 0.001 --dropout 0.1 --l2 1e-4 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.0 > output_reddit_rgat_true
python main_cold_relational_gnn.py  --data_name reddit  --input_size 602 --gnn_model RGAT  --lr 0.001 --dropout 0.1 --l2 1e-4 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 > output_reddit_rgat_25_edge
python main_cold_relational_gnn.py  --data_name reddit  --input_size 602 --gnn_model RGAT  --lr 0.001 --dropout 0.1 --l2 1e-4 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 --blind node > output_reddit_rgat_25_node
python main_cold_relational_gnn.py  --data_name reddit  --input_size 602 --gnn_model RGAT  --lr 0.001 --dropout 0.1 --l2 1e-4 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.50 > output_reddit_rgat_50_edge
python main_cold_relational_gnn.py  --data_name reddit  --input_size 602 --gnn_model RGAT  --lr 0.001 --dropout 0.1 --l2 1e-4 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.50 --blind node > output_reddit_rgat_50_node
python main_cold_relational_gnn.py  --data_name reddit  --input_size 602 --gnn_model RGAT  --lr 0.001 --dropout 0.1 --l2 1e-4 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 > output_reddit_rgat_75_edge
python main_cold_relational_gnn.py  --data_name reddit  --input_size 602 --gnn_model RGAT  --lr 0.001 --dropout 0.1 --l2 1e-4 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 --blind node > output_reddit_rgat_75_node
python main_cold_relational_gnn.py  --data_name reddit  --input_size 602 --gnn_model RGAT  --lr 0.001 --dropout 0.1 --l2 1e-4 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.90 > output_reddit_rgat_90_edge
python main_cold_relational_gnn.py  --data_name reddit  --input_size 602 --gnn_model RGAT  --lr 0.001 --dropout 0.1 --l2 1e-4 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.90 --blind node > output_reddit_rgat_90_node

#SAGE
python main_cold_gnn.py  --data_name reddit  --input_size 602 --gnn_model SAGE  --lr 0.001 --dropout 0.5 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.0 > output_reddit_sage_true
python main_cold_gnn.py  --data_name reddit  --input_size 602 --gnn_model SAGE  --lr 0.001 --dropout 0.5 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 > output_reddit_sage_25_edge
python main_cold_gnn.py  --data_name reddit  --input_size 602 --gnn_model SAGE  --lr 0.001 --dropout 0.5 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 --blind node > output_reddit_sage_25_node
python main_cold_gnn.py  --data_name reddit  --input_size 602 --gnn_model SAGE  --lr 0.001 --dropout 0.5 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.50 > output_reddit_sage_50_edge
python main_cold_gnn.py  --data_name reddit  --input_size 602 --gnn_model SAGE  --lr 0.001 --dropout 0.5 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.50 --blind node > output_reddit_sage_50_node
python main_cold_gnn.py  --data_name reddit  --input_size 602 --gnn_model SAGE  --lr 0.001 --dropout 0.5 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 > output_reddit_sage_75_edge
python main_cold_gnn.py  --data_name reddit  --input_size 602 --gnn_model SAGE  --lr 0.001 --dropout 0.5 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 --blind node > output_reddit_sage_75_node
python main_cold_gnn.py  --data_name reddit  --input_size 602 --gnn_model SAGE  --lr 0.001 --dropout 0.5 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.90 > output_reddit_sage_90_edge
python main_cold_gnn.py  --data_name reddit  --input_size 602 --gnn_model SAGE  --lr 0.001 --dropout 0.5 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.90 --blind node > output_reddit_sage_90_node

#GAE
python main_cold_gae.py  --data_name reddit  --input_size 602 --gnn_model GCN  --with_loss_weight --lr 0.001 --dropout 0.3 --l2 1e-7 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.0 > output_reddit_gae_true
python main_cold_gae.py  --data_name reddit  --input_size 602 --gnn_model GCN  --with_loss_weight --lr 0.001 --dropout 0.3 --l2 1e-7 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 > output_reddit_gae_25_edge
python main_cold_gae.py  --data_name reddit  --input_size 602 --gnn_model GCN  --with_loss_weight --lr 0.001 --dropout 0.3 --l2 1e-7 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 --blind node > output_reddit_gae_25_node
python main_cold_gae.py  --data_name reddit  --input_size 602 --gnn_model GCN  --with_loss_weight --lr 0.001 --dropout 0.3 --l2 1e-7 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.50 > output_reddit_gae_50_edge
python main_cold_gae.py  --data_name reddit  --input_size 602 --gnn_model GCN  --with_loss_weight --lr 0.001 --dropout 0.3 --l2 1e-7 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.50 --blind node > output_reddit_gae_50_node
python main_cold_gae.py  --data_name reddit  --input_size 602 --gnn_model GCN  --with_loss_weight --lr 0.001 --dropout 0.3 --l2 1e-7 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 > output_reddit_gae_75_edge
python main_cold_gae.py  --data_name reddit  --input_size 602 --gnn_model GCN  --with_loss_weight --lr 0.001 --dropout 0.3 --l2 1e-7 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 --blind node > output_reddit_gae_75_node
python main_cold_gae.py  --data_name reddit  --input_size 602 --gnn_model GCN  --with_loss_weight --lr 0.001 --dropout 0.3 --l2 1e-7 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.90 > output_reddit_gae_90_edge
python main_cold_gae.py  --data_name reddit  --input_size 602 --gnn_model GCN  --with_loss_weight --lr 0.001 --dropout 0.3 --l2 1e-7 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.90 --blind node > output_reddit_gae_90_node

#mlp_model
python main_cold_gnn.py  --data_name reddit  --input_size 602 --gnn_model mlp_model  --lr 0.001 --dropout 0.3 --l2 1e-4 --num_layers 2 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.0 > output_reddit_mlp_true

#MF
python main_cold_gnn.py  --data_name reddit  --input_size 602 --gnn_model MF --max_nodes 247 --lr 0.001 --dropout 0.3 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.0 > output_reddit_mf_true

#NeoGNN
python main_cold_neognn.py  --data_name reddit  --input_size 602 --gnn_model NeoGNN  --lr 0.001 --dropout 0.3 --l2 1e-7 --num_layers 1 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.0 > output_reddit_neognn_true
python main_cold_neognn.py  --data_name reddit  --input_size 602 --gnn_model NeoGNN  --lr 0.001 --dropout 0.3 --l2 1e-7 --num_layers 1 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 > output_reddit_neognn_25_edge
python main_cold_neognn.py  --data_name reddit  --input_size 602 --gnn_model NeoGNN  --lr 0.001 --dropout 0.3 --l2 1e-7 --num_layers 1 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 --blind node > output_reddit_neognn_25_node
python main_cold_neognn.py  --data_name reddit  --input_size 602 --gnn_model NeoGNN  --lr 0.001 --dropout 0.3 --l2 1e-7 --num_layers 1 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.50 > output_reddit_neognn_50_edge
python main_cold_neognn.py  --data_name reddit  --input_size 602 --gnn_model NeoGNN  --lr 0.001 --dropout 0.3 --l2 1e-7 --num_layers 1 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.50 --blind node > output_reddit_neognn_50_node
python main_cold_neognn.py  --data_name reddit  --input_size 602 --gnn_model NeoGNN  --lr 0.001 --dropout 0.3 --l2 1e-7 --num_layers 1 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 > output_reddit_neognn_75_edge
python main_cold_neognn.py  --data_name reddit  --input_size 602 --gnn_model NeoGNN  --lr 0.001 --dropout 0.3 --l2 1e-7 --num_layers 1 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 --blind node > output_reddit_neognn_75_node
python main_cold_neognn.py  --data_name reddit  --input_size 602 --gnn_model NeoGNN  --lr 0.001 --dropout 0.3 --l2 1e-7 --num_layers 1 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.90 > output_reddit_neognn_90_edge
python main_cold_neognn.py  --data_name reddit  --input_size 602 --gnn_model NeoGNN  --lr 0.001 --dropout 0.3 --l2 1e-7 --num_layers 1 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.90 --blind node > output_reddit_neognn_90_node

#NCN
python main_cold_ncn.py  --dataset reddit  --input_size 602  --gnnlr 0.001 --prelr 0.001 --l2 1e-4  --predp 0.3 --gnndp 0.3  --mplayers 1 --nnlayers 2 --hiddim 128 --testbs 512 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --ln --lnnn --predictor cn1 --runs 10 --model puregcn --maskinput --jk --use_xlin --tailact --epochs 9999 --kill_cnt 3 --eval_steps 5 --cold_perc 0.0 > output_reddit_ncn_true
python main_cold_ncn.py  --dataset reddit  --input_size 602  --gnnlr 0.001 --prelr 0.001 --l2 1e-4  --predp 0.3 --gnndp 0.3  --mplayers 1 --nnlayers 2 --hiddim 128 --testbs 512 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --ln --lnnn --predictor cn1 --runs 10 --model puregcn --maskinput --jk --use_xlin --tailact --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 > output_reddit_ncn_25_edge
python main_cold_ncn.py  --dataset reddit  --input_size 602  --gnnlr 0.001 --prelr 0.001 --l2 1e-4  --predp 0.3 --gnndp 0.3  --mplayers 1 --nnlayers 2 --hiddim 128 --testbs 512 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --ln --lnnn --predictor cn1 --runs 10 --model puregcn --maskinput --jk --use_xlin --tailact --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 --blind node > output_reddit_ncn_25_node
python main_cold_ncn.py  --dataset reddit  --input_size 602  --gnnlr 0.001 --prelr 0.001 --l2 1e-4  --predp 0.3 --gnndp 0.3  --mplayers 1 --nnlayers 2 --hiddim 128 --testbs 512 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --ln --lnnn --predictor cn1 --runs 10 --model puregcn --maskinput --jk --use_xlin --tailact --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.50 > output_reddit_ncn_50_edge
python main_cold_ncn.py  --dataset reddit  --input_size 602  --gnnlr 0.001 --prelr 0.001 --l2 1e-4  --predp 0.3 --gnndp 0.3  --mplayers 1 --nnlayers 2 --hiddim 128 --testbs 512 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --ln --lnnn --predictor cn1 --runs 10 --model puregcn --maskinput --jk --use_xlin --tailact --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.50 --blind node > output_reddit_ncn_50_node
python main_cold_ncn.py  --dataset reddit  --input_size 602  --gnnlr 0.001 --prelr 0.001 --l2 1e-4  --predp 0.3 --gnndp 0.3  --mplayers 1 --nnlayers 2 --hiddim 128 --testbs 512 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --ln --lnnn --predictor cn1 --runs 10 --model puregcn --maskinput --jk --use_xlin --tailact --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 > output_reddit_ncn_75_edge
python main_cold_ncn.py  --dataset reddit  --input_size 602  --gnnlr 0.001 --prelr 0.001 --l2 1e-4  --predp 0.3 --gnndp 0.3  --mplayers 1 --nnlayers 2 --hiddim 128 --testbs 512 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --ln --lnnn --predictor cn1 --runs 10 --model puregcn --maskinput --jk --use_xlin --tailact --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 --blind node > output_reddit_ncn_75_node
python main_cold_ncn.py  --dataset reddit  --input_size 602  --gnnlr 0.001 --prelr 0.001 --l2 1e-4  --predp 0.3 --gnndp 0.3  --mplayers 1 --nnlayers 2 --hiddim 128 --testbs 512 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --ln --lnnn --predictor cn1 --runs 10 --model puregcn --maskinput --jk --use_xlin --tailact --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.90 > output_reddit_ncn_90_edge
python main_cold_ncn.py  --dataset reddit  --input_size 602  --gnnlr 0.001 --prelr 0.001 --l2 1e-4  --predp 0.3 --gnndp 0.3  --mplayers 1 --nnlayers 2 --hiddim 128 --testbs 512 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --ln --lnnn --predictor cn1 --runs 10 --model puregcn --maskinput --jk --use_xlin --tailact --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.90 --blind node > output_reddit_ncn_90_node
