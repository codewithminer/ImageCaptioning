import pickle

def getSceneGraphData():
    objects = []
    with (open("Detected_Scene_Graphs_SUN397_1.pkl", "rb")) as f:
        objects.append(pickle.load(f))
    node_labels = []
    edge_labels = []
    edge_index = []
    for ind,obj in enumerate(objects[0]):
        node_labels.append(objects[0][obj]['node_labels'])
        edge_labels.append(objects[0][obj]['edge_labels'])
        edge_index.append(objects[0][obj]['edges'])
    return node_labels, edge_labels, edge_index

def makeAdjMatrix(edge_index):
    E = edge_index
    # print(E)
    # nodes must be numbers in a sequential range starting at 0 - so this is the
    # number of nodes. you can assert this is the case as well if desired 
    size = len(set([n for e in E for n in e])) 
    # make an empty adjacency list  
    adjacency = [[0]*size for _ in range(size)]
    # populate the list for each edge
    for sink, source in E:
        adjacency[sink][source] = 1
    return adjacency