import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import json
from os import path
from sklearn.metrics import confusion_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from matplotlib import pyplot as plt


# Consider the edge case - what happens if we have no FP and non-zero FN and vice-versa.
# The matrix should include a slack code for this - currently resolved with default slack code 99

# Keep the ANDs or not? there could be a correlation?
# Keeping ANDs in results in a co-occurrence matrix (co-occurrence between predictions and gold standard)

def load_translation_dict_from_icd9(fn_icd9_graph_json='ICD9/icd9_graph_desc.json'):
    """Load the icd9 graph translation dictionary"""
    with open(path.abspath(fn_icd9_graph_json),encoding='utf-8') as json_file:    
        translation_dict_icd9 = json.load(json_file)
    return translation_dict_icd9

def confusion_mat_st(pred_v, gold_v, filter_ands = True, slack_variable = 9000):
    rows = []
    cols = []
    vals = []
    left = pred_v
    right = gold_v
    if filter_ands:
        ands = pred_v & gold_v # finding True Positives
        ands_converted = np.where(ands == 1)[0] # determining locations of ands
        left = (pred_v - ands)
        right = gold_v - ands
        rows = list(ands_converted) # populating the matrix with the true positives (row indices)
        cols = list(ands_converted) # populating the matrix with the true positives (column indices)
        vals = list(np.ones(shape=ands_converted.shape, dtype="int32")) # populating the matrix with the true positives
    left_converted = np.where(left == 1)[0] # conversion to index form from than binary form
    right_converted = np.where(right == 1)[0] # conversion to indes form from the binary form
    if len(left_converted) == 0 and len(right_converted) == 0:
        # This option means that there are no lefts/rights left. Either there were no predictions and no gold standard labels
        # or, if we are filtering ANDs, for this doc we have perfect precision and recall
        return rows, cols, vals
    # Should there be leftovers in only one of the sets (predictions/expectation), we create a slack (OOF) code to associate with
    # meaning that no relevant related code was around to be checked against, so the misprediction could be attributed to
    # a different family altogether. However, this ASSUMES that all true positives were only representing a single respective correct prediction
    # that was corresponded to a single gold label - it could be that for a single gold label the model learns to predict 2 associated labels,
    # one correctly and one incorrectly. 
    elif len(left_converted) == 0:
        left_converted = [slack_variable] 
    elif len(right_converted) == 0:
        right_converted = [slack_variable]

    # Associating leftovers in each set with one another - each FP (pred) with each FN (gold label)
    for i in left_converted:
        for j in right_converted:
            rows.append(i)
            cols.append(j)
            # If scaling is to be added, this is the place to put it.
            val = 1
            vals.append(val)
    return rows, cols, vals
    
def apply_filters(results, filter_array):
    """
    Takes an array of predictions or gold standard and applies filters on it (in our case ancestor-filters).
    We start with a 2D matrix representing an array (one dim) of positive predictions/labels for each document (other dim).
    The number of filters is the number of ancestor level codes of interest in our application.
    The result is a 3D tensor, where the one dim is the prediction-level labels, one dim is the documents, and final dim is the filter.
    """
    filtered_arrays = []
    for f in filter_array:
        filtered_arrays.append((results.T*f).T)
    return np.array(filtered_arrays)
    
def apply_filters_smart(results, filter_array):
    """
    This function is a more efficient implementation of the apply_filters method
    """
    len_filters = len(filter_array)
    stacked_results = np.stack([results]*len_filters)
    reshaped_results = stacked_results.swapaxes(0,2).swapaxes(1,2)
    filtered_reshaped_results = reshaped_results*filter_array
    filtered_reshaped_back = filtered_reshaped_results.swapaxes(1,2).swapaxes(0,2)
    # return (stacked_results.swapaxes(0,2).swapaxes(1,2)*filter_array).swapaxes(1,2).swapaxes(0,2)
    return filtered_reshaped_back
    
def roll_single_filter(filtered_pred, filtered_gold, filter_ands = True, oof_id = 9000):
    r,c,v = [],[],[]
    assert filtered_pred.shape == filtered_gold.shape
    for i in range(len(filtered_pred)):
        new_r, new_c, new_v = confusion_mat_st(filtered_pred[i], filtered_gold[i], filter_ands, slack_variable = oof_id)
        r += new_r
        c += new_c
        v += new_v
    return r,c,v
    
def unravel_rolled_filter(r,c,v, oof_id = 9000):
    counter = defaultdict(int)
    for i in range(len(r)):
        counter[(r[i], c[i])]+=v[i]
    rows = [oof_id]
    cols = [oof_id]
    vals = [0]
    
    for key in counter.keys():
        rows.append(key[0])
        cols.append(key[1])
        vals.append(counter[key])
    return rows, cols, vals
    
def roll_and_unravel_filter(filtered_pred, filtered_gold, filter_ands = True, oof_id = 9000):
    r,c,v = roll_single_filter(filtered_pred, filtered_gold, filter_ands, oof_id) # it seems the conf_matrix operation is not working properly here
    return unravel_rolled_filter(r,c,v, oof_id)
    
def roll_and_unravel_all_filters(filtered_pred_tensor, filtered_gold_tensor, filter_ands = True, oof_id = 9000):
    filtered_csr_data = []
    for i in range(filtered_pred_tensor.shape[0]):
        filtered_csr_data.append(roll_and_unravel_filter(filtered_pred_tensor[i].T, filtered_gold_tensor[i].T, filter_ands, oof_id))
    return filtered_csr_data

def combine_csr_matrices(csr_def_list, oof_id = 9000):
    counter = defaultdict(int)
    rows = []
    cols = []
    vals = []
    for triplet in csr_def_list:
        r,c,v = triplet
        rx, cx, vx = unravel_rolled_filter(r,c,v, oof_id)
        rows += rx
        cols += cx
        vals += vx
    return rows, cols, vals

def produce_conf_matrix(pred, gold, filter_ands = True, oof_id = 9000):
    unraveled = roll_and_unravel_filter(pred.T, gold.T, filter_ands, oof_id)
    c_mat = csr_matrix((unraveled[2],(unraveled[0],unraveled[1])))
    return c_mat
    
def produce_all_conf_matrices(preds, golds, filter_ands = True, oof_id = 9000):
    pairs = list(zip(preds, golds))
    c_mats = []
    for pair in pairs:
        p, g = pair
        c_mats.append(produce_conf_matrix(p,g, filter_ands, oof_id))
    return c_mats
    
def analyse_conf_matrix(c_mat):
    ys = np.where(c_mat.max(0).toarray()>0)[1] # determine which relevant gold standard values appear with non-zero count
    xs = []
    results = [] # index of original diagonal value, said value, index of max value, max value
    for y in ys:
        x = c_mat[:, y].argmax() # find the index at which said gold standard value has the highest count in confusion
        y_val = c_mat[y,y] # determine the diagonal value (TP)
        full_sum = (np.sum(c_mat[:,y]))
        xs.append(x)
        x_val = c_mat[x,y] # determine the value for the maximum
        oof = c_mat[-1,y]
        oof_percent =  round(oof/full_sum, 3)
        recall_percent = round(y_val/full_sum, 3)
        results.append((y, y_val, recall_percent, x, x_val, round(x_val/full_sum, 3), oof_percent, round(1 - recall_percent - oof_percent,3)))
    return results
    
def analyse_all_conf_matrices(c_mats):
    pairs = []
    for c_mat in c_mats:
        pairs += analyse_conf_matrix(c_mat)
    return pairs
    
def translate_ind_to_code(pairs, ind2c):
    code_pairs = []
    for pair in pairs:
        x_id, x_count, x_pc, y_id, y_count, y_pc, oof_pc, ifc_pc = pair
        code_pairs.append((ind2c[x_id],x_count, x_pc, ind2c[y_id], y_count, y_pc, oof_pc, ifc_pc))
    return code_pairs
    
def convert_into_df(tuples):
    df=pd.DataFrame(tuples, columns = ["Identity Code","Identity Count", "Identity Percentage", "Preferred Prediction Code", "Preferred Prediction Count", "Preferred Prediction Percentage", "OOF-Percentage", "In-Family-Confusion Percentage"])
    df["Match"] = df["Identity Code"]==df["Preferred Prediction Code"]
    return df
    
    
def describe_families(family_filters, code_dict_reverse):
    code_lists  = []
    print("call")
    with open("families.txt", "w") as f:
        for a in family_filters:
            id_list = list(np.where(a==1)[0])
            code_list = [code_dict_reverse[c] for c in id_list]
            print(code_list)
            code_lists.append(str(code_list)+"\n")
        print(code_lists)
        f.writelines(code_lists)
        
def run(preds, golds, family_filters, code_dict_reverse, filter_ands = True, oof_id = 9000):
    # Data Setup
    #print(f"preds shape: {preds.T.shape}; family_filters shape: {family_filters.T.shape}")
    #print(preds.T.shape, golds.T.shape, family_filters.T.shape)
    filtered_preds = apply_filters_smart(preds.T, family_filters.T)
    #print(f"filtered_preds: {filtered_preds.shape}")
    filtered_golds = apply_filters_smart(golds.T, family_filters.T)
    #print(f"filtered_golds: {filtered_golds.shape}")
    all_conf_matrices = produce_all_conf_matrices(filtered_preds, filtered_golds, filter_ands = filter_ands, oof_id = oof_id)
    #print(f"All confusion matrices: {np.array(all_conf_matrices).shape}")
    result=translate_ind_to_code(analyse_all_conf_matrices(all_conf_matrices), code_dict_reverse)
    return convert_into_df(result), all_conf_matrices
    
def coocurrence(preds, golds, family_filters, code_dict_reverse, oof_id = 9000):
    result, mats = run(preds, golds, family_filters, code_dict_reverse, False, oof_id)
    filtered_result = result[result["Identity Code"] !="OOF"]
    return filtered_result, mats
    
def hierarchical_confusion_unscaled(preds, golds, family_filters, code_dict_reverse, oof_id = 9000):
    result, mats = run(preds, golds, family_filters, code_dict_reverse, True, oof_id)
    filtered_result = result[result["Identity Code"] !="OOF"]
    return filtered_result, mats
    
def hierarchical_confusion_unscaled_one(preds, golds, family_filters, code_dict_reverse, i=0, oof_id = 9000):
    fltr = family_filters.T[i]
    result, mats = run(preds, golds,  fltr.reshape(fltr.shape[0],1), code_dict_reverse, True, oof_id)
    result = result.drop_duplicates()
    result["Num Known Family Codes"] = sum(family_filters.T[i])
    filtered_result = result[result["Identity Code"] !="OOF"]
    return filtered_result, mats

def hierarchical_confusion_unscaled_all(preds, golds, family_filters, code_dict_reverse, oof_id = 9000):
    filters = family_filters.shape[1]
    results = []
    all_mats = []
    for i in tqdm(range(filters)):
        fltr = family_filters.T[i]
        result, mats = run(preds, golds, fltr.reshape(fltr.shape[0],1), code_dict_reverse, True, oof_id)
        result["Num Known Family Codes"] = sum(family_filters.T[i])
        results.append(result)
        all_mats+=mats
    results = pd.concat(results).drop_duplicates()
    filtered_result = results[results["Identity Code"] !="OOF"]
    return filtered_result, all_mats
    
def analyse_results(df):
    mean_correct_prediction  = np.mean(df["Identity Percentage"])
    mean_conf_prediction  = np.mean(df["In-Family-Confusion Percentage"])
    mean_oof_prediction  = np.mean(df["OOF-Percentage"])
    match_percentage =  np.mean(df["Match"])
    result_dict = dict({"mean correct prediction %": mean_correct_prediction,
                        "mean within-family-confusion %": mean_conf_prediction,
                        "mean oof confusion %": mean_oof_prediction,
                        "identity-preferred match percentage %": match_percentage})
    return result_dict


def visualiseHCM(cm, ind_to_code, name):
    inds = sorted(list(set(np.where(cm>0)[0]).union(set(np.where(cm>0)[1]))))
    rows = []
    codes = [ind_to_code[ind] for ind in inds]
    sorted_codes_reindex = np.argsort(codes)
    inds = list(np.array(inds)[sorted_codes_reindex])
    codes = list(np.array(codes)[sorted_codes_reindex])
    if codes[0] == 'OOF':
    	inds = inds[1:]+[inds[0]]
    	codes = codes[1:]+[codes[0]]
        
    for x_i in inds:
        row = []
        for y_i in inds:
            row.append(cm[x_i,y_i])
        rows.append(row)
    reduced_matrix = np.array(rows)
    return visualiseMat(reduced_matrix, codes, name)
    
def family_index(code, c_dict, filters):
    fam = np.where(filters[c_dict[code]]==1)[0][0]
    return fam

def visualiseMat(hcm, codes, name):
    plt.clf()
    plt.imshow(hcm, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto')
    codes_sans_oof = list(codes).copy()
    codes_sans_oof.remove('OOF')
    code_fam = codes_sans_oof[0].split(".")[0]
    plt.title(f'Confusion Matrix for Code Family {code_fam}')
    plt.ylabel('Predicted Label')
    plt.xlabel('True Label')
       
    thresh = hcm.max() / 2.
    for i in range(hcm.shape[0]):
        for j in range(hcm.shape[1]):
            plt.text(j, i, format(hcm[i, j]),
                    ha="center", va="center",
                    color="white" if  hcm[i, j] > thresh else "black") 
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(len(codes))
    plt.xticks(tick_marks, codes, rotation=45)
    plt.yticks(tick_marks, codes)
    plt.autoscale()
    return plt.savefig( f"result_matrix_images/{name}_{code_fam}"+".jpg", format = "jpg", bbox_inches='tight')

if __name__=="__main__":
    #data_read
    preds = pd.read_csv("sample_predictions.csv").drop(columns=["Unnamed: 0"])
    golds = pd.read_csv("sample_gold.csv").drop(columns=["Unnamed: 0"])
    
    
    #sample filter_setup
    if 'Unnamed: 0' in preds.columns:
    	preds = preds.drop(columns=['Unnamed: 0'])
    
    if 'Unnamed: 0' in golds.columns:
    	golds = golds.drop(columns=['Unnamed: 0'])
    codes = list(preds.columns)
    code_dict = dict(zip(codes, range(len(codes))))
    code_dict_r = dict({code_dict[code]:code for code in code_dict.keys()})
    
    
    icd9_dict = load_translation_dict_from_icd9()
    
    all_codes_list = list(codes)
    all_codes_parent_list = ([icd9_dict[code]['parents'][1] for code in all_codes_list])
    
    parent_list = list(set(all_codes_parent_list))
    rows = []
    cols = []
    for i, label_i in enumerate (all_codes_parent_list):
        for j, label_j in enumerate (parent_list):
            if label_i == label_j:
                rows.append(i)
                cols.append(j)
    
    vals = [1]*len(rows)
    mat = coo_matrix((vals, (rows, cols)))
    family_filters = mat.toarray() 
    
    preds_converted = (preds>=0.5)*1
    golds_converted = (golds>0)*1
    slack_code = len(codes)
    code_dict_r[slack_code]="OOF"
    
    """
    co_results = coocurrence(preds_converted, golds_converted, family_filters, code_dict_r, oof_id = slack_code)
    co_results.to_csv("pred_gold_coocurrence.csv", index=False)
    print("vanilla")
    hi_results = hierarchical_confusion_unscaled(preds_converted, golds_converted, family_filters, code_dict_r, oof_id = slack_code)
    hi_results.to_csv("pred_gold_hierarchical_confusion.csv", index=False)
    print("transposed")
    hi_results_T = hierarchical_confusion_unscaled(golds_converted, preds_converted, family_filters, code_dict_r, oof_id = slack_code)
    hi_results_T.to_csv("gold_pred_hierarchical_confusion.csv", index=False)
    print("iterative")
    """
    hi_results_T_0, ms = hierarchical_confusion_unscaled_one(golds_converted, preds_converted, family_filters, code_dict_r, oof_id = slack_code)
    hi_results_T_0.to_csv("gold_pred_hierarchical_confusion_0.csv", index=False)
    visualiseHCM(ms[0].todense(), code_dict_r, "dummy") 
    fam = family_index("401.9", code_dict, family_filters)
    print(fam)
    
    hi_results_T_all, a_ms = hierarchical_confusion_unscaled_all(golds_converted, preds_converted, family_filters, code_dict_r, oof_id = slack_code)
    hi_results_T_all.to_csv("gold_pred_hierarchical_confusion_all.csv", index=False)
    
    visualiseHCM(a_ms[fam].todense(), code_dict_r, "401.9")
    for fam in range(len(a_ms)):
        visualiseHCM(a_ms[fam].todense(), code_dict_r, f"fam")
    
    hi_results_0, ms = hierarchical_confusion_unscaled_one(preds_converted, golds_converted, family_filters, code_dict_r, oof_id = slack_code)
    hi_results_0.to_csv("pred_gold_hierarchical_confusion_0.csv", index=False)
    hi_results_all, ms = hierarchical_confusion_unscaled_all(preds_converted, golds_converted, family_filters, code_dict_r, oof_id = slack_code)
    hi_results_all.to_csv("pred_gold_hierarchical_confusion_all.csv", index=False)
    
    describe_families(family_filters.T, code_dict_r)
    
    print(analyse_results(hi_results_all))
