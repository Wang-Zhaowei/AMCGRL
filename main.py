from parms_setting import settings
import load_data
from sklearn.model_selection import KFold
import numpy as np
import random
import Model
from sklearn.ensemble import RandomForestClassifier
import test_scores as score

def Main():
    args = settings()
    file_path = './Dataset/' + str(args.inter_type) + '/'
    mole_x = load_data.load_moles(file_path+args.input_x_file)
    mole_y = load_data.load_moles(file_path+args.input_y_file)
    inter_file = file_path+'molecular interacting pairs.txt'
    non_inter_file = file_path+'molecular non-interacting pairs.txt'
    x_seq_file = file_path + args.input_x_file[:-5] + ' sequences.fasta'
    y_seq_file = file_path + args.input_y_file[:-5] + ' sequences.fasta'

    x_feat_dict = {}
    y_feat_dict = {}
    if args.inter_type == 'AA':
        x_feat_dict = load_data.load_pseaac_feat(x_seq_file, mole_x)
        y_feat_dict = load_data.load_pseaac_feat(y_seq_file, mole_y)
    elif args.inter_type == 'NN':
        x_feat_dict = load_data.load_psednc_feat(x_seq_file, mole_x)
        y_feat_dict = load_data.load_psednc_feat(y_seq_file, mole_y)
    elif args.inter_type == 'NA':
        x_feat_dict = load_data.load_psednc_feat(x_seq_file, mole_x)
        y_feat_dict = load_data.load_pseaac_feat(y_seq_file, mole_y)
    elif args.inter_type == 'DT':
        x_feat_dict = load_data.load_drug_feat(x_seq_file, mole_x)
        y_feat_dict = load_data.load_pseaac_feat(y_seq_file, mole_y)

    x_seq_mat = load_data.cal_seq_sim(x_feat_dict)
    y_seq_mat = load_data.cal_seq_sim(y_feat_dict)
    pairs = load_data.load_pairs(inter_file, non_inter_file, mole_x, mole_y)
    random.shuffle(pairs)
    pairs = np.array(pairs)

    eval_metrics = []
    n_fold = 5
    Kfold = KFold(n_splits=n_fold, shuffle=True)
    for train_index,test_index in Kfold.split(pairs):
        train_pairs, test_pairs = pairs[train_index], pairs[test_index]
        train_adj = load_data.cal_adj_mat(train_pairs, len(mole_x), len(mole_y))
        x_ip_mat = load_data.cal_ip_sim(train_adj)
        x_sim_mat = args.alpha * x_seq_mat + (1 - args.alpha) * x_ip_mat
        y_ip_mat = load_data.cal_ip_sim(train_adj.T)
        y_sim_mat = args.beta * y_seq_mat + (1 - args.beta) * y_ip_mat
        print(x_sim_mat.shape, y_sim_mat.shape)

        model = Model.AMCGRL(args)
        train_data = load_data.dataset(x_sim_mat, y_sim_mat, train_pairs, x_feat_dict, y_feat_dict)

        print("Feature representation learning ...\n")
        x_feat, y_feat = Model.feature_representation(model, len(mole_x), len(mole_y), train_data, args)
        data_train, y_train = load_data.final_dataset(x_feat, y_feat, train_pairs)
        data_test, y_test = load_data.final_dataset(x_feat, y_feat, test_pairs)

        print("Binary classification ...\n")
        clf = RandomForestClassifier()
        clf.fit(data_train, y_train)
        y_prob = clf.predict_proba(data_test)[:,-1]
        
        tp, fp, tn, fn, acc, prec, recall, MCC, f1_score, AUC, AUPR = score.calculate_performace(y_prob, y_test)
        eval_metrics.append([tp, fp, tn, fn, acc, prec, recall, MCC, f1_score, AUC, AUPR])
        print('\ntp = \t'+ str(tp) + '\t fp = \t'+ str(fp) + '\t tn = \t'+ str(tn)+ '\t fn = \t'+ str(fn)+'\n  Acc = \t', acc, '\n  prec = \t', prec, '\n  recall = \t', recall, '\n  MCC = \t', MCC, '\n  f1_score = \t', f1_score, '\n  AUC = \t', AUC)

    ave_tp, ave_fp, ave_tn, ave_fn, ave_acc, ave_prec, ave_recall, ave_MCC, ave_f1_score, ave_AUC, ave_AUPR = score.get_average_metrics(eval_metrics)
    print('\n Acc = \t'+ str(ave_acc)+'\n prec = \t'+ str(ave_prec)+ '\n recall = \t'+str(ave_recall)+ '\n MCC = \t'+str(ave_MCC)+'\n f1_score = \t'+str(ave_f1_score)+'\n AUC = \t'+ str(ave_AUC) + '\n AUPR =\t'+str(ave_AUPR)+'\n')
    fw = open('./Results/'+args.inter_type+' '+args.out_file, 'a+')
    fw.write('tp\t'+str(ave_tp)+'\tfp\t'+str(ave_fp)+'\ttn\t'+str(ave_tn)+'\tfn\t'+str(ave_fn)+'\tAcc\t'+str(ave_acc)+'\tPrec\t'+str(ave_prec)+'\tRec\t'+str(ave_recall)+'\tMCC\t'+str(ave_MCC)+'\tF1\t'+str(ave_f1_score)+'\tAUC\t'+str(ave_AUC)+'\tAUPR\t'+str(ave_AUPR)+'\n')

if __name__ == "__main__":
    Main()