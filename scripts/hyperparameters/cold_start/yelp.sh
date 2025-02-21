cd benchmarking/cold_start

## Leroy
python main_cold_leroy.py  --data_name yelp  --model Leroy --epochs 1 --kill_cnt 1 --eval_steps 1  --runs 1 --cold_perc 0.0 > output_yelp_leroy_true

## ACCSLP
python main_cold_accslp.py  --data_name yelp  --model ACCSLP  --alpha 20 --beta 27 --rank 5 --groups 100 --max_nodes 264 --epochs 1 --kill_cnt 1 --eval_steps 1  --runs 1 --cold_perc 0.0 > output_yelp_accslp_true
python main_cold_accslp.py  --data_name yelp  --model ACCSLP  --alpha 20 --beta 27 --rank 5 --groups 100 --max_nodes 264 --epochs 1 --kill_cnt 1 --eval_steps 1  --runs 1 --cold_perc 0.25 > output_yelp_accslp_25_edge
python main_cold_accslp.py  --data_name yelp  --model ACCSLP  --alpha 20 --beta 27 --rank 5 --groups 100 --max_nodes 264 --epochs 1 --kill_cnt 1 --eval_steps 1  --runs 1 --cold_perc 0.25 --blind node > output_yelp_accslp_25_node
python main_cold_accslp.py  --data_name yelp  --model ACCSLP  --alpha 20 --beta 27 --rank 5 --groups 100 --max_nodes 264 --epochs 1 --kill_cnt 1 --eval_steps 1  --runs 1 --cold_perc 0.50 > output_yelp_accslp_50_edge
python main_cold_accslp.py  --data_name yelp  --model ACCSLP  --alpha 20 --beta 27 --rank 5 --groups 100 --max_nodes 264 --epochs 1 --kill_cnt 1 --eval_steps 1  --runs 1 --cold_perc 0.50 --blind node > output_yelp_accslp_50_node
python main_cold_accslp.py  --data_name yelp  --model ACCSLP  --alpha 20 --beta 27 --rank 5 --groups 100 --max_nodes 264 --epochs 1 --kill_cnt 1 --eval_steps 1  --runs 1 --cold_perc 0.75 > output_yelp_accslp_75_edge
python main_cold_accslp.py  --data_name yelp  --model ACCSLP  --alpha 20 --beta 27 --rank 5 --groups 100 --max_nodes 264 --epochs 1 --kill_cnt 1 --eval_steps 1  --runs 1 --cold_perc 0.75 --blind node > output_yelp_accslp_75_node
python main_cold_accslp.py  --data_name yelp  --model ACCSLP  --alpha 20 --beta 27 --rank 5 --groups 100 --max_nodes 264 --epochs 1 --kill_cnt 1 --eval_steps 1  --runs 1 --cold_perc 0.90 > output_yelp_accslp_90_edge
python main_cold_accslp.py  --data_name yelp  --model ACCSLP  --alpha 20 --beta 27 --rank 5 --groups 100 --max_nodes 264 --epochs 1 --kill_cnt 1 --eval_steps 1  --runs 1 --cold_perc 0.90 --blind node > output_yelp_accslp_90_node

## CN
python main_cold_heuristic.py --data_name yelp --use_heuristic CN --cold_perc 0.0 > output_yelp_cn_true
python main_cold_heuristic.py --data_name yelp --use_heuristic CN --cold_perc 0.25 > output_yelp_cn_25_edge
python main_cold_heuristic.py --data_name yelp --use_heuristic CN --cold_perc 0.25 --blind node > output_yelp_cn_25_node
python main_cold_heuristic.py --data_name yelp --use_heuristic CN --cold_perc 0.50 > output_yelp_cn_50_edge
python main_cold_heuristic.py --data_name yelp --use_heuristic CN --cold_perc 0.50 --blind node > output_yelp_cn_50_node
python main_cold_heuristic.py --data_name yelp --use_heuristic CN --cold_perc 0.75 > output_yelp_cn_75_edge
python main_cold_heuristic.py --data_name yelp --use_heuristic CN --cold_perc 0.75 --blind node > output_yelp_cn_75_node
python main_cold_heuristic.py --data_name yelp --use_heuristic CN --cold_perc 0.90 > output_yelp_cn_90_edge
python main_cold_heuristic.py --data_name yelp --use_heuristic CN --cold_perc 0.90 --blind node > output_yelp_cn_90_node

## AA
python main_cold_heuristic.py --data_name yelp --use_heuristic AA --cold_perc 0.0 > output_yelp_aa_true
python main_cold_heuristic.py --data_name yelp --use_heuristic AA --cold_perc 0.25 > output_yelp_aa_25_edge
python main_cold_heuristic.py --data_name yelp --use_heuristic AA --cold_perc 0.25 --blind node > output_yelp_aa_25_node
python main_cold_heuristic.py --data_name yelp --use_heuristic AA --cold_perc 0.50 > output_yelp_aa_50_edge
python main_cold_heuristic.py --data_name yelp --use_heuristic AA --cold_perc 0.50 --blind node > output_yelp_aa_50_node
python main_cold_heuristic.py --data_name yelp --use_heuristic AA --cold_perc 0.75 > output_yelp_aa_75_edge
python main_cold_heuristic.py --data_name yelp --use_heuristic AA --cold_perc 0.75 --blind node > output_yelp_aa_75_node
python main_cold_heuristic.py --data_name yelp --use_heuristic AA --cold_perc 0.90 > output_yelp_aa_90_edge
python main_cold_heuristic.py --data_name yelp --use_heuristic AA --cold_perc 0.90 --blind node > output_yelp_aa_90_node

## RA
python main_cold_heuristic.py --data_name yelp --use_heuristic RA --cold_perc 0.0 > output_yelp_ra_true
python main_cold_heuristic.py --data_name yelp --use_heuristic RA --cold_perc 0.25 > output_yelp_ra_25_edge
python main_cold_heuristic.py --data_name yelp --use_heuristic RA --cold_perc 0.25 --blind node > output_yelp_ra_25_node
python main_cold_heuristic.py --data_name yelp --use_heuristic RA --cold_perc 0.50 > output_yelp_ra_50_edge
python main_cold_heuristic.py --data_name yelp --use_heuristic RA --cold_perc 0.50 --blind node > output_yelp_ra_50_node
python main_cold_heuristic.py --data_name yelp --use_heuristic RA --cold_perc 0.75 > output_yelp_ra_75_edge
python main_cold_heuristic.py --data_name yelp --use_heuristic RA --cold_perc 0.75 --blind node > output_yelp_ra_75_node
python main_cold_heuristic.py --data_name yelp --use_heuristic RA --cold_perc 0.90 > output_yelp_ra_90_edge
python main_cold_heuristic.py --data_name yelp --use_heuristic RA --cold_perc 0.90 --blind node > output_yelp_ra_90_node

## shortest path
python main_cold_heuristic.py --data_name yelp --use_heuristic shortest_path --cold_perc 0.0 > output_yelp_shortest_true
python main_cold_heuristic.py --data_name yelp --use_heuristic shortest_path --cold_perc 0.25 > output_yelp_shortest_25_edge
python main_cold_heuristic.py --data_name yelp --use_heuristic shortest_path --cold_perc 0.25 --blind node > output_yelp_shortest_25_node
python main_cold_heuristic.py --data_name yelp --use_heuristic shortest_path --cold_perc 0.50 > output_yelp_shortest_50_edge
python main_cold_heuristic.py --data_name yelp --use_heuristic shortest_path --cold_perc 0.50 --blind node > output_yelp_shortest_50_node
python main_cold_heuristic.py --data_name yelp --use_heuristic shortest_path --cold_perc 0.75 > output_yelp_shortest_75_edge
python main_cold_heuristic.py --data_name yelp --use_heuristic shortest_path --cold_perc 0.75 --blind node > output_yelp_shortest_75_node
python main_cold_heuristic.py --data_name yelp --use_heuristic shortest_path --cold_perc 0.90 > output_yelp_shortest_90_edge
python main_cold_heuristic.py --data_name yelp --use_heuristic shortest_path --cold_perc 0.90 --blind node > output_yelp_shortest_90_node

## katz
python main_cold_heuristic.py --data_name yelp --use_heuristic katz_close --cold_perc 0.0 > output_yelp_katz_true
python main_cold_heuristic.py --data_name yelp --use_heuristic katz_close --cold_perc 0.25 > output_yelp_katz_25_edge
python main_cold_heuristic.py --data_name yelp --use_heuristic katz_close --cold_perc 0.25 --blind node > output_yelp_katz_25_node
python main_cold_heuristic.py --data_name yelp --use_heuristic katz_close --cold_perc 0.50 > output_yelp_katz_50_edge
python main_cold_heuristic.py --data_name yelp --use_heuristic katz_close --cold_perc 0.50 --blind node > output_yelp_katz_50_node
python main_cold_heuristic.py --data_name yelp --use_heuristic katz_close --cold_perc 0.75 > output_yelp_katz_75_edge
python main_cold_heuristic.py --data_name yelp --use_heuristic katz_close --cold_perc 0.75 --blind node > output_yelp_katz_75_node
python main_cold_heuristic.py --data_name yelp --use_heuristic katz_close --cold_perc 0.90 > output_yelp_katz_90_edge
python main_cold_heuristic.py --data_name yelp --use_heuristic katz_close --cold_perc 0.90 --blind node > output_yelp_katz_90_node

#GCN
python main_cold_gnn.py  --data_name yelp  --input_size 300 --gnn_model GCN  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.0 > output_yelp_gcn_true
python main_cold_gnn.py  --data_name yelp  --input_size 300 --gnn_model GCN  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 > output_yelp_gcn_25_edge
python main_cold_gnn.py  --data_name yelp  --input_size 300 --gnn_model GCN  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 --blind node > output_yelp_gcn_25_node
python main_cold_gnn.py  --data_name yelp  --input_size 300 --gnn_model GCN  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.50 > output_yelp_gcn_50_edge
python main_cold_gnn.py  --data_name yelp  --input_size 300 --gnn_model GCN  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.50 --blind node > output_yelp_gcn_50_node
python main_cold_gnn.py  --data_name yelp  --input_size 300 --gnn_model GCN  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 > output_yelp_gcn_75_edge
python main_cold_gnn.py  --data_name yelp  --input_size 300 --gnn_model GCN  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 --blind node > output_yelp_gcn_75_node
python main_cold_gnn.py  --data_name yelp  --input_size 300 --gnn_model GCN  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.90 > output_yelp_gcn_90_edge
python main_cold_gnn.py  --data_name yelp  --input_size 300 --gnn_model GCN  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.90 --blind node > output_yelp_gcn_90_node

#RGCN
python main_cold_relational_gnn.py  --data_name yelp  --input_size 300 --gnn_model RGCN  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.0 > output_yelp_rgcn_true
python main_cold_relational_gnn.py  --data_name yelp  --input_size 300 --gnn_model RGCN  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 > output_yelp_rgcn_25_edge
python main_cold_relational_gnn.py  --data_name yelp  --input_size 300 --gnn_model RGCN  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 --blind node > output_yelp_rgcn_25_node
python main_cold_relational_gnn.py  --data_name yelp  --input_size 300 --gnn_model RGCN  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.50 > output_yelp_rgcn_50_edge
python main_cold_relational_gnn.py  --data_name yelp  --input_size 300 --gnn_model RGCN  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.50 --blind node > output_yelp_rgcn_50_node
python main_cold_relational_gnn.py  --data_name yelp  --input_size 300 --gnn_model RGCN  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 > output_yelp_rgcn_75_edge
python main_cold_relational_gnn.py  --data_name yelp  --input_size 300 --gnn_model RGCN  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 --blind node > output_yelp_rgcn_75_node
python main_cold_relational_gnn.py  --data_name yelp  --input_size 300 --gnn_model RGCN  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.90 > output_yelp_rgcn_90_edge
python main_cold_relational_gnn.py  --data_name yelp  --input_size 300 --gnn_model RGCN  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.90 --blind node > output_yelp_rgcn_90_node

#GAT
python main_cold_gnn.py  --data_name yelp  --input_size 300 --gnn_model GAT  --lr 0.001 --dropout 0.1 --l2 0 --num_layers 1 --hidden_channels 256  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.0 > output_yelp_gat_true
python main_cold_gnn.py  --data_name yelp  --input_size 300 --gnn_model GAT  --lr 0.001 --dropout 0.1 --l2 0 --num_layers 1 --hidden_channels 256  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 > output_yelp_gat_25_edge
python main_cold_gnn.py  --data_name yelp  --input_size 300 --gnn_model GAT  --lr 0.001 --dropout 0.1 --l2 0 --num_layers 1 --hidden_channels 256  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 --blind node > output_yelp_gat_25_node
python main_cold_gnn.py  --data_name yelp  --input_size 300 --gnn_model GAT  --lr 0.001 --dropout 0.1 --l2 0 --num_layers 1 --hidden_channels 256  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.50 > output_yelp_gat_50_edge
python main_cold_gnn.py  --data_name yelp  --input_size 300 --gnn_model GAT  --lr 0.001 --dropout 0.1 --l2 0 --num_layers 1 --hidden_channels 256  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.50 --blind node > output_yelp_gat_50_node
python main_cold_gnn.py  --data_name yelp  --input_size 300 --gnn_model GAT  --lr 0.001 --dropout 0.1 --l2 0 --num_layers 1 --hidden_channels 256  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 > output_yelp_gat_75_edge
python main_cold_gnn.py  --data_name yelp  --input_size 300 --gnn_model GAT  --lr 0.001 --dropout 0.1 --l2 0 --num_layers 1 --hidden_channels 256  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 --blind node > output_yelp_gat_75_node
python main_cold_gnn.py  --data_name yelp  --input_size 300 --gnn_model GAT  --lr 0.001 --dropout 0.1 --l2 0 --num_layers 1 --hidden_channels 256  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.90 > output_yelp_gat_90_edge
python main_cold_gnn.py  --data_name yelp  --input_size 300 --gnn_model GAT  --lr 0.001 --dropout 0.1 --l2 0 --num_layers 1 --hidden_channels 256  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.90 --blind node > output_yelp_gat_90_node

#SAGE
python main_cold_gnn.py  --data_name yelp  --input_size 300 --gnn_model SAGE  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 3 --hidden_channels 256  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.0 > output_yelp_sage_true
python main_cold_gnn.py  --data_name yelp  --input_size 300 --gnn_model SAGE  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 3 --hidden_channels 256  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 > output_yelp_sage_25_edge
python main_cold_gnn.py  --data_name yelp  --input_size 300 --gnn_model SAGE  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 3 --hidden_channels 256  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 --blind node > output_yelp_sage_25_node
python main_cold_gnn.py  --data_name yelp  --input_size 300 --gnn_model SAGE  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 3 --hidden_channels 256  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.50 > output_yelp_sage_50_edge
python main_cold_gnn.py  --data_name yelp  --input_size 300 --gnn_model SAGE  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 3 --hidden_channels 256  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.50 --blind node > output_yelp_sage_50_node
python main_cold_gnn.py  --data_name yelp  --input_size 300 --gnn_model SAGE  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 3 --hidden_channels 256  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 > output_yelp_sage_75_edge
python main_cold_gnn.py  --data_name yelp  --input_size 300 --gnn_model SAGE  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 3 --hidden_channels 256  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 --blind node > output_yelp_sage_75_node
python main_cold_gnn.py  --data_name yelp  --input_size 300 --gnn_model SAGE  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 3 --hidden_channels 256  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.90 > output_yelp_sage_90_edge
python main_cold_gnn.py  --data_name yelp  --input_size 300 --gnn_model SAGE  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 3 --hidden_channels 256  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.90 --blind node > output_yelp_sage_90_node

#GAE
python main_cold_gae.py  --data_name yelp  --input_size 300 --gnn_model GCN  --with_loss_weight --lr 0.001 --dropout 0.1 --l2 0 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.0 > output_yelp_gae_true
python main_cold_gae.py  --data_name yelp  --input_size 300 --gnn_model GCN  --with_loss_weight --lr 0.001 --dropout 0.1 --l2 0 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 > output_yelp_gae_25_edge
python main_cold_gae.py  --data_name yelp  --input_size 300 --gnn_model GCN  --with_loss_weight --lr 0.001 --dropout 0.1 --l2 0 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 --blind node > output_yelp_gae_25_node
python main_cold_gae.py  --data_name yelp  --input_size 300 --gnn_model GCN  --with_loss_weight --lr 0.001 --dropout 0.1 --l2 0 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.50 > output_yelp_gae_50_edge
python main_cold_gae.py  --data_name yelp  --input_size 300 --gnn_model GCN  --with_loss_weight --lr 0.001 --dropout 0.1 --l2 0 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.50 --blind node > output_yelp_gae_50_node
python main_cold_gae.py  --data_name yelp  --input_size 300 --gnn_model GCN  --with_loss_weight --lr 0.001 --dropout 0.1 --l2 0 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 > output_yelp_gae_75_edge
python main_cold_gae.py  --data_name yelp  --input_size 300 --gnn_model GCN  --with_loss_weight --lr 0.001 --dropout 0.1 --l2 0 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 --blind node > output_yelp_gae_75_node
python main_cold_gae.py  --data_name yelp  --input_size 300 --gnn_model GCN  --with_loss_weight --lr 0.001 --dropout 0.1 --l2 0 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.90 > output_yelp_gae_90_edge
python main_cold_gae.py  --data_name yelp  --input_size 300 --gnn_model GCN  --with_loss_weight --lr 0.001 --dropout 0.1 --l2 0 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.90 --blind node > output_yelp_gae_90_node

#mlp_model
python main_cold_gnn.py  --data_name yelp  --input_size 300 --gnn_model mlp_model  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 3 --hidden_channels 256  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.0 > output_yelp_mlp_true

#MF
python main_cold_gnn.py  --data_name yelp  --input_size 300 --gnn_model MF --max_nodes 264  --lr 0.001 --dropout 0.1 --l2 0 --num_layers 1 --hidden_channels 256  --num_layers_predictor 3  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.0 > output_yelp_mf_true

#NeoGNN
python main_cold_neognn.py  --data_name yelp  --input_size 300 --gnn_model NeoGNN  --lr 0.01 --dropout 0.3 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.0 > output_yelp_neognn_true
python main_cold_neognn.py  --data_name yelp  --input_size 300 --gnn_model NeoGNN  --lr 0.01 --dropout 0.3 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 > output_yelp_neognn_25_edge
python main_cold_neognn.py  --data_name yelp  --input_size 300 --gnn_model NeoGNN  --lr 0.01 --dropout 0.3 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 --blind node > output_yelp_neognn_25_node
python main_cold_neognn.py  --data_name yelp  --input_size 300 --gnn_model NeoGNN  --lr 0.01 --dropout 0.3 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.50 > output_yelp_neognn_50_edge
python main_cold_neognn.py  --data_name yelp  --input_size 300 --gnn_model NeoGNN  --lr 0.01 --dropout 0.3 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.50 --blind node > output_yelp_neognn_50_node
python main_cold_neognn.py  --data_name yelp  --input_size 300 --gnn_model NeoGNN  --lr 0.01 --dropout 0.3 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 > output_yelp_neognn_75_edge
python main_cold_neognn.py  --data_name yelp  --input_size 300 --gnn_model NeoGNN  --lr 0.01 --dropout 0.3 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 --blind node > output_yelp_neognn_75_node
python main_cold_neognn.py  --data_name yelp  --input_size 300 --gnn_model NeoGNN  --lr 0.01 --dropout 0.3 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.90 > output_yelp_neognn_90_edge
python main_cold_neognn.py  --data_name yelp  --input_size 300 --gnn_model NeoGNN  --lr 0.01 --dropout 0.3 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.90 --blind node > output_yelp_neognn_90_node

#NCN
python main_cold_ncn.py  --dataset yelp  --input_size 300  --gnnlr 0.01 --prelr 0.01 --l2 1e-7  --predp 0.1 --gnndp 0.1  --mplayers 1 --nnlayers 2 --hiddim 128 --testbs 512 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --ln --lnnn --predictor cn1 --runs 10 --model puregcn --maskinput --jk --use_xlin --tailact --epochs 9999 --kill_cnt 3 --eval_steps 5 --cold_perc 0.0 > output_yelp_ncn_true
python main_cold_ncn.py  --dataset yelp  --input_size 300  --gnnlr 0.01 --prelr 0.01 --l2 1e-7  --predp 0.1 --gnndp 0.1  --mplayers 1 --nnlayers 2 --hiddim 128 --testbs 512 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --ln --lnnn --predictor cn1 --runs 10 --model puregcn --maskinput --jk --use_xlin --tailact --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 > output_yelp_ncn_25_edge
python main_cold_ncn.py  --dataset yelp  --input_size 300  --gnnlr 0.01 --prelr 0.01 --l2 1e-7  --predp 0.1 --gnndp 0.1  --mplayers 1 --nnlayers 2 --hiddim 128 --testbs 512 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --ln --lnnn --predictor cn1 --runs 10 --model puregcn --maskinput --jk --use_xlin --tailact --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 --blind node > output_yelp_ncn_25_node
python main_cold_ncn.py  --dataset yelp  --input_size 300  --gnnlr 0.01 --prelr 0.01 --l2 1e-7  --predp 0.1 --gnndp 0.1  --mplayers 1 --nnlayers 2 --hiddim 128 --testbs 512 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --ln --lnnn --predictor cn1 --runs 10 --model puregcn --maskinput --jk --use_xlin --tailact --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.50 > output_yelp_ncn_50_edge
python main_cold_ncn.py  --dataset yelp  --input_size 300  --gnnlr 0.01 --prelr 0.01 --l2 1e-7  --predp 0.1 --gnndp 0.1  --mplayers 1 --nnlayers 2 --hiddim 128 --testbs 512 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --ln --lnnn --predictor cn1 --runs 10 --model puregcn --maskinput --jk --use_xlin --tailact --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.50 --blind node > output_yelp_ncn_50_node
python main_cold_ncn.py  --dataset yelp  --input_size 300  --gnnlr 0.01 --prelr 0.01 --l2 1e-7  --predp 0.1 --gnndp 0.1  --mplayers 1 --nnlayers 2 --hiddim 128 --testbs 512 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --ln --lnnn --predictor cn1 --runs 10 --model puregcn --maskinput --jk --use_xlin --tailact --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 > output_yelp_ncn_75_edge
python main_cold_ncn.py  --dataset yelp  --input_size 300  --gnnlr 0.01 --prelr 0.01 --l2 1e-7  --predp 0.1 --gnndp 0.1  --mplayers 1 --nnlayers 2 --hiddim 128 --testbs 512 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --ln --lnnn --predictor cn1 --runs 10 --model puregcn --maskinput --jk --use_xlin --tailact --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 --blind node > output_yelp_ncn_75_node
python main_cold_ncn.py  --dataset yelp  --input_size 300  --gnnlr 0.01 --prelr 0.01 --l2 1e-7  --predp 0.1 --gnndp 0.1  --mplayers 1 --nnlayers 2 --hiddim 128 --testbs 512 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --ln --lnnn --predictor cn1 --runs 10 --model puregcn --maskinput --jk --use_xlin --tailact --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.90 > output_yelp_ncn_90_edge
python main_cold_ncn.py  --dataset yelp  --input_size 300  --gnnlr 0.01 --prelr 0.01 --l2 1e-7  --predp 0.1 --gnndp 0.1  --mplayers 1 --nnlayers 2 --hiddim 128 --testbs 512 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --ln --lnnn --predictor cn1 --runs 10 --model puregcn --maskinput --jk --use_xlin --tailact --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.90 --blind node > output_yelp_ncn_90_node
