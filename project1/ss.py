from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import igraph as ig
import numpy as np
from pykeen.datasets import get_dataset

datas = ["FB15k","FB15k237","WN18","WN18RR","YAGO310"]

for data in datas :

    if data != "FB15k237" :
        dataset = get_dataset(dataset=data)
        path = dataset.training.metadata['path']
    else :
        path = '/home/kang/.data/pykeen/datasets/fb15k237/train.txt'
    

    # training set
    table_m = pd.read_csv(path, sep='\t')
    table_m.columns = ["out", "rel", "inn"]

    # sort entity
    table_cnt_in = table_m.groupby("inn").size().reset_index(name="n").sort_values(by="n", ascending=False)
    table_cnt_out = table_m.groupby("out").size().reset_index(name="n").sort_values(by="n", ascending=False)
    table_cnt_all = pd.merge(table_cnt_out, table_cnt_in, left_on="out", right_on="inn", how="inner")
    table_cnt_all["n"] = table_cnt_all["n_x"] + table_cnt_all["n_y"]
    df = table_cnt_all[["out", "n"]].rename(columns={"out": "entity"}).sort_values(by="n", ascending=False)

    entity_num = len(table_cnt_all)

    # test set
    if data != "FB15k237" :
        path = dataset.testing.metadata['path']
    else :
        path = '/home/kang/.data/pykeen/datasets/fb15k237/test.txt'
        
    table_t = pd.read_csv(path, sep='\t', header=None)
    table_t.columns = ["out", "rel", "inn"]

    test_triple_num = len(table_t)

    # variables setup
    removed_triple_from_test_top = []
    removed_triple_from_test_bottom = []
    removed_triple_from_test_both = []

    percent_T = [0] # 0 # 1
    percent_T += [0.1*(2*i+1) for i in range(5)] # 0.1 ~ 0.9 # 5
    percent_T += [2*i+1 for i in range(5)] # 1 3 5 7 9 # 5
    percent_T += [(i+2)*5 for i in range(5)] # 10 15 20 25 30 # 5
    percent_B = [(i)*10 for i in range(8)] # 0 10 20 30 40 50 60 70 # 8

    removed_entity_head = [int((i)/100*entity_num) for i in percent_T]
    removed_entity_tail = [int((i)/100*entity_num) for i in percent_B]

    # for head

    # for num in removed_entity_head:

    #     remove_entity = []
        
    #     for n in range(num):
    #         remove_entity.append(df.head(num).iat[n,0]) #tail for bottom

    #     for v in remove_entity:
    #         table_t = table_t[table_t.out != v]
    #         table_t = table_t[table_t.inn != v]

    #     removed_triple_from_test_top.append(int(len(table_t)/test_triple_num*100))

    # # for tail

    # table_t = pd.read_csv(path, sep='\t', header=None)
    # table_t.columns = ["out", "rel", "inn"]

    # for num in removed_entity_tail:

    #     remove_entity = []
        
    #     for n in range(num):
    #         remove_entity.append(df.tail(num).iat[n,0]) #tail for bottom

    #     for v in remove_entity:
    #         table_t = table_t[table_t.out != v]
    #         table_t = table_t[table_t.inn != v]

    #     removed_triple_from_test_bottom.append(int(len(table_t)/test_triple_num*100))
    
    # for both

    heatmap = pd.DataFrame(index = removed_entity_head, columns = removed_entity_tail)

    for numh in removed_entity_head:
        
        table_t = pd.read_csv(path, sep='\t', header=None)
        table_t.columns = ["out", "rel", "inn"]
        
        remove_entity = []
        
        for n in range(numh):
            remove_entity.append(df.head(numh).iat[n,0])
        
        for numt in removed_entity_tail:
            
            for n in range(numt):
                remove_entity.append(df.tail(numt).iat[n,0])

            for v in remove_entity:
                table_t = table_t[table_t.out != v]
                table_t = table_t[table_t.inn != v]

            heatmap.loc[[numh], [numt]] = (int(len(table_t)/test_triple_num*100))
    
    # graph

    # plt.figure(figsize = (12,8))

    # df1=pd.DataFrame({'X':percent_T,'Y':removed_triple_from_test_top})
    # plt.subplot(2, 1, 1)
    # plt.plot(df1['X'],df1['Y'])
    # plt.title('percent of triple removed from testset [top entity]')

    # df2=pd.DataFrame({'X':percent_B,'Y':removed_triple_from_test_bottom})
    # plt.subplot(2, 1, 2)
    # plt.plot(df2['X'],df2['Y'])
    # plt.title('percent of triple removed from testset [bottom entity]')

    # plt.suptitle(data+' INDIVIDUAL REMOVAL')
    # #plt.show()
    # plt.savefig(data+' INDIVIDUAL REMOVAL')
    

    heatmap.index = percent_T
    heatmap.columns = percent_B

    heatmap = heatmap.astype('float')
    print(heatmap.dtypes)
    print(heatmap)

    
    sns.heatmap(heatmap, annot=True, fmt='f')
    plt.title('percent of triple removed from testset', fontsize=20)

    plt.figure(figsize = (15,15))

    plt.suptitle(data+' COMBINED REMOVAL')
    #plt.show()
    plt.savefig(data+' COMBINED REMOVAL')

