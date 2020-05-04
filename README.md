# Neural_Decoder
 
This repository includes the source code for the paper "Neural Decoder based Linear Graph Autoencoder for Graph Embedding"

There are three tasks:
(1) unipartite link prediction (cora, citeseer, pubmed)
(2) bipartite link prediction ( GPC, Enzymes, Ionchannel, Malaria, Drug-target, SW, Na-net, Movielens)
(3) node cluserting (cora, citeseer)

To run the algorithm,
 for task (1), first choose the dataset in args.py within unipartite_link_prediction folder, and execute "python neural_decoder.py"
 for task (2), first choose the dataset in args.py within bipartite_link_prediction folder, and execute "python neural_decoder.py"
 for task (3) , first choose the dataset in args.py within node_clustering folder, and execute "python train.py"
