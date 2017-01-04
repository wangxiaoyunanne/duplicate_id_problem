import re
import urllib2
import pandas as pd
import igraph
import networkx as nx
from networkx.algorithms import bipartite

response = urllib2.urlopen('http://cyrus.cs.ucdavis.edu/sfelixwu/Syria/name-id-pages/')
html = response.read()
names_pos = [m.start() for m in re.finditer('name00000000', html)]
txt_pos =  [m.start() for m in re.finditer('.txt', html)]
url_list = []
output_list = []
# get url list 
for i in range (0, len(names_pos)) :
    if (i%2) :
        url_list.append('http://cyrus.cs.ucdavis.edu/sfelixwu/Syria/name-id-pages/' + html [names_pos[i] : (txt_pos[i]+4)])
        #output_list.append('output/' + html [names_pos[i] : (txt_pos[i]+4)])
# for each url get data
for url in url_list :
    data = urllib2.urlopen(url)
    data = data.readlines()
    data_df = pd.DataFrame(index = range(0, len(data)), columns = ['user','page'])
    for j in range(0, len(data)):
        data_df.ix[j] = data[j].split()
#    data_tp = list(data_df.itertuples(index = False))
    data_user = data_df['user']
#    data_page = data_df['page']
    # after get data frame 
    # construct the bipartite graph
    g = nx.from_pandas_dataframe(data_df, 'user', 'page')
    #g.add_nodes_from(data_user, bipartite =0)
    #g.add_nodes_from(data_page, bipartite =1)
    g_proj =   bipartite.projected_graph(g,data_user , multigraph=True)
    nb_comp = nx.number_connected_components(g_proj)
    id_group = sorted(nx.connected_components(g_proj), key = len, reverse=True)

    output_file = 'output/'+ url[57:]
    with open (output_file, "w") as text_file :
        text_file.write ("number of components:{} ".format(nb_comp))
        text_file.write("\n")
        for k in range(0, nb_comp) :
            id_gk = id_group[k]
            text_file.write("{}".format(id_gk))
            text_file.write("\n")
    text_file.close()

