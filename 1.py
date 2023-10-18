import os


os.system("PGD/pgd -f " + "data/edgelist.txt" + " -o natural --micro " + "data/motif_list.txt > " + "data/motif_list_process.txt")