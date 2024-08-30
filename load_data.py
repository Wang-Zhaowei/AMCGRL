import numpy as np
import math
import torch
import pickle
import itertools


def load_moles(data_file):
	moles = []
	with open(data_file, 'r') as fa:
		for mole in fa:
			moles.append(mole[:-1])
	return moles


def load_pairs(posi_file, nega_file, mole_x, mole_y):
	posi_pairs = []
	with open(posi_file, 'r') as fa:
		for line in fa:
			x, y = line.split('\t')
			posi_pairs.append([mole_x.index(x), mole_y.index(y[:-1]), 1])

	nega_pairs = []
	with open(nega_file, 'r') as fa:
		for line in fa:
			x, y = line.split('\t')
			nega_pairs.append([mole_x.index(x), mole_y.index(y[:-1]), 0])
	return posi_pairs+nega_pairs


def Rvalue(aa1, aa2, AADict, Matrix):
	return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)
	

def calculate_PseAAC(sequence, lambdaValue=4, w=0.05):
	dataFile = './Dataset/PAAC.txt'
	with open(dataFile) as f:
		records = f.readlines()
	AA = ''.join(records[0].rstrip().split()[1:])
	AADict = {}
	for i in range(len(AA)):
		AADict[AA[i]] = i
	AAProperty = []
	AAPropertyNames = []
	for i in range(1, len(records)):
		array = records[i].rstrip().split() if records[i].rstrip() != '' else None
		AAProperty.append([float(j) for j in array[1:]])
		AAPropertyNames.append(array[0])

	AAProperty1 = []
	for i in AAProperty:
		meanI = sum(i) / 20
		fenmu = math.sqrt(sum([(j-meanI)**2 for j in i])/20)
		AAProperty1.append([(j-meanI)/fenmu for j in i])

	encodings = []
	header = ['#']
	for aa in AA:
		header.append('Xc1.' + aa)
	for n in range(1, lambdaValue + 1):
		header.append('Xc2.lambda' + str(n))
	encodings.append(header)

	code = []
	theta = []
	for n in range(1, lambdaValue + 1):
		theta.append(
			sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in range(len(sequence) - n)]) / (
			len(sequence) - n))
	myDict = {}
	for aa in AA:
		myDict[aa] = sequence.count(aa)
	code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
	code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
	return code


def get_kmer_frequency(sequence, kmer, rna_comb):
    myFrequency = {}
    for pep in rna_comb:
        myFrequency[pep] = 0
    for i in range(len(sequence) - kmer + 1):
        myFrequency[sequence[i: i + kmer]] = myFrequency[sequence[i: i + kmer]] + 1
    for key in myFrequency:
        myFrequency[key] = myFrequency[key] / (len(sequence) - kmer + 1)
    return myFrequency


def correlationFunction(pepA, pepB, myIndex, myPropertyName, myPropertyValue):
    CC = 0
    for p in myPropertyName:
        CC = CC + (float(myPropertyValue[p][myIndex.index(pepA)]) - float(myPropertyValue[p][myIndex.index(pepB)])) ** 2
    return CC / len(myPropertyName)


def get_theta_array(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, kmer):
    thetaArray = []
    for tmpLamada in range(lamadaValue):
        theta = 0
        for i in range(len(sequence) - tmpLamada - kmer):
            theta = theta + correlationFunction(sequence[i:i + kmer],
                                                sequence[i + tmpLamada + 1: i + tmpLamada + 1 + kmer], myIndex,
                                                myPropertyName, myPropertyValue)
        thetaArray.append(theta / (len(sequence) - tmpLamada - kmer))
    return thetaArray


def calculate_PseDNC(sequence, lamada=2, weight=0.1):
	baseSymbol = 'ACGU'
	rna_comb = [''.join(i) for i in list(itertools.product(baseSymbol, repeat=2))]

	with open('./Dataset/dirnaPhyche.data', 'rb') as f:
		Property_value = pickle.load(f)
	property_index =  ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']

	code = []
	dipeptideFrequency = get_kmer_frequency(sequence, 2, rna_comb)
	thetaArray = get_theta_array(rna_comb, property_index, Property_value, lamada, sequence, 2)
	for pair in rna_comb:
		code.append(dipeptideFrequency[pair] / (1 + weight * sum(thetaArray)))
	for k in range(17, 16 + lamada + 1):
		code.append((weight * thetaArray[k - 17]) / (1 + weight * sum(thetaArray)))
	return code


def load_pseaac_feat(seq_file, moles):
	seq_dict = {}
	with open(seq_file, 'r') as rf:
		seq = ''
		for line in rf:
			line = line.strip()
			if line[0] == '>':
				name = line[1:]
			else:
				seq = line.upper()
				pseaac = calculate_PseAAC(seq)
				seq_dict[moles.index(name)] = pseaac
	return seq_dict


def load_psednc_feat(seq_file, moles):
	seq_dict = {}
	with open(seq_file, 'r') as rf:
		seq = ''
		for line in rf:
			line = line.strip()
			if line[0] == '>':
				name = line[1:]
			else:
				seq = line.upper().replace("T","U")
				psednc = calculate_PseDNC(seq)
				if name in moles:
					seq_dict[moles.index(name)] = psednc
	return seq_dict


def cal_seq_sim(seq_dict):
	seq_sim = np.zeros([len(seq_dict), len(seq_dict)])
	for i in seq_dict:
		for j in seq_dict:
			vec_i = np.array(seq_dict[i])
			vec_j = np.array(seq_dict[j])
			seq_sim[i, j] = vec_i.dot(vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j))
	return seq_sim


def cal_ip_sim(adj):
    width = 0
    for i in range(adj.shape[0]):
        width += np.sum(adj[i])

    GIPK_sim = np.zeros([adj.shape[0], adj.shape[0]])
    for i in range(adj.shape[0]):
        for j in range(adj.shape[0]):
            GIPK_sim[i, j] = math.exp((np.sum((adj[i] - adj[j])**2) * (1/(width/adj.shape[0]))) * (-1))
    return GIPK_sim


def cal_adj_mat(pairs, num_x, num_y):
	adj = np.zeros(shape = (num_x, num_y), dtype=float)
	for pair in pairs:
		x, y, label = pair
		adj[x, y] = label
	return adj


def get_edge_index(matrix):
	edge_index = [[], []]
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			if matrix[i][j] != 0:
				edge_index[0].append(i)
				edge_index[1].append(j)
	return torch.LongTensor(edge_index)


def dataset(x_sim_mat, y_sim_mat, pairs, x_feat_dict, y_feat_dict):
	dataset = {}
	x_feat = []
	for i in range((len(x_feat_dict))):
		x_feat.append(x_feat_dict[i])
	x_edge_index = get_edge_index(x_sim_mat)
	dataset['X'] = {'attribute': torch.from_numpy(np.array(x_feat)).float(), 'matrix': torch.from_numpy(x_sim_mat).float(), 'edges': x_edge_index}
	
	y_edge_index = get_edge_index(y_sim_mat)
	y_feat = []
	for i in range((len(y_feat_dict))):
		y_feat.append(y_feat_dict[i])
	dataset['Y'] = {'attribute': torch.from_numpy(np.array(y_feat)).float(), 'matrix': torch.from_numpy(y_sim_mat).float(), 'edges': y_edge_index}

	Matrix = np.zeros(shape = (x_sim_mat.shape[0]+y_sim_mat.shape[0], x_sim_mat.shape[0]+y_sim_mat.shape[0]), dtype=float)
	for pair in pairs:
		x, y, label = pair
		Matrix[x, y+x_sim_mat.shape[0]] = label
		Matrix[y+x_sim_mat.shape[0], x] = label
	inter_edge_index = get_edge_index(Matrix)
	dataset['Inter'] = inter_edge_index
	
	Adj = np.zeros(shape = (x_sim_mat.shape[0], y_sim_mat.shape[0]), dtype=float)
	for pair in pairs:
		x, y, label = pair
		Adj[x, y] = label
	dataset['A'] = torch.from_numpy(Adj).float()
	return dataset


def final_dataset(x_feat, y_feat, pairs):
	data = []
	labels = []
	for pair in pairs:
		x, y, label = pair
		data.append(x_feat[x,:].tolist() + y_feat[y,:].tolist())
		labels.append(label)
	return np.array(data), np.array(labels)
