import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix



# Consider the edge case - what happens if we have no FP and non-zero FN and vice-versa.
# The matrix should include a slack code for this - currently resolved with default slack code 99

# Keep the ANDs or not? there could be a correlation?
# Keeping ANDs in results in a co-occurrence matrix (co-occurrence between predictions and gold standard)

def confusion_mat_st(pred_v, gold_v, filter_ands = True, slack_variable = 99):
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
    
def roll_single_filter(filtered_pred, filtered_gold, filter_ands = True):
    r,c,v = [],[],[]
    assert filtered_pred.shape == filtered_gold.shape
    for i in range(len(filtered_pred)):
        new_r, new_c, new_v = confusion_mat_st(filtered_pred[i], filtered_gold[i], filter_ands)
        r += new_r
        c += new_c
        v += new_v
    return r,c,v
    
def unravel_rolled_filter(r,c,v):
    counter = defaultdict(int)
    for i in range(len(r)):
        counter[(r[i], c[i])]+=v[i]
    rows = [99]
    cols = [99]
    vals = [0]
    
    for key in counter.keys():
        rows.append(key[0])
        cols.append(key[1])
        vals.append(counter[key])
    return rows, cols, vals
    
def roll_and_unravel_filter(filtered_pred, filtered_gold, filter_ands = True):
    r,c,v = roll_single_filter(filtered_pred, filtered_gold, filter_ands) # it seems the conf_matrix operation is not working properly here
    return unravel_rolled_filter(r,c,v)
    
def roll_and_unravel_all_filters(filtered_pred_tensor, filtered_gold_tensor, filter_ands = True):
    filtered_csr_data = []
    for i in range(filtered_pred_tensor.shape[0]):
        filtered_csr_data.append(roll_and_unravel_filter(filtered_pred_tensor[i].T, filtered_gold_tensor[i].T, filter_ands))
    return filtered_csr_data

def combine_csr_matrices(csr_def_list):
    counter = defaultdict(int)
    rows = []
    cols = []
    vals = []
    for triplet in csr_def_list:
        r,c,v = triplet
        rx, cx, vx = unravel_rolled_filter(r,c,v)
        rows += rx
        cols += cx
        vals += vx
    return rows, cols, vals

def produce_conf_matrix(pred, gold, filter_ands = True):
    unraveled = roll_and_unravel_filter(pred.T, gold.T, filter_ands)
    # print(f"unraveled: {unraveled}")
    c_mat = csr_matrix((unraveled[2],(unraveled[0],unraveled[1])))
    return c_mat
    
def produce_all_conf_matrices(preds, golds, filter_ands = True):
    pairs = list(zip(preds, golds))
    c_mats = []
    for pair in pairs:
        p, g = pair
        c_mats.append(produce_conf_matrix(p,g, filter_ands))
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
        results.append((y, y_val, y_val/full_sum, x, x_val, x_val/full_sum))
    return results
    
def analyse_all_conf_matrices(c_mats):
    pairs = []
    for c_mat in c_mats:
        pairs += analyse_conf_matrix(c_mat)
    return pairs
    
def translate_ind_to_code(pairs, ind2c):
    code_pairs = []
    for pair in pairs:
        x_id, x_count, x_pc, y_id, y_count, y_pc = pair
        code_pairs.append((ind2c[x_id],x_count, x_pc, ind2c[y_id], y_count, y_pc))
    return code_pairs
    
def convert_into_df(tuples):
    return pd.DataFrame(tuples, columns = ["Identity Code","Identity Count", "Identity Percentage", "Preferred Prediction Code", "Preferred Prediction Count", "Preferred Prediction Percentage"])

def run(preds, golds, family_filters, code_dict_reverse, filter_ands = True):
    # Data Setup
    print(preds.T.shape, golds.T.shape, family_filters.T.shape)
    filtered_preds = apply_filters_smart(preds.T, family_filters.T)
    print(f"filtered_preds: {filtered_preds.shape}")
    filtered_golds = apply_filters_smart(golds.T, family_filters.T)
    print(f"filtered_golds: {filtered_golds.shape}")
    all_conf_matrices = produce_all_conf_matrices(filtered_preds, filtered_golds, filter_ands = filter_ands)
    print(f"All confusion matrices: {np.array(all_conf_matrices).shape}")
    result=translate_ind_to_code(analyse_all_conf_matrices(all_conf_matrices), code_dict_reverse)
    return convert_into_df(result)
    
def coocurrence(preds, golds, family_filters, code_dict_reverse):
    result = run(preds, golds, family_filters, code_dict_reverse, False)
    filtered_result = result[result["Identity Code"] !="Slack"]
    return filtered_result
    
def hierarchical_confusion_unscaled(preds, golds, family_filters, code_dict_reverse):
    result = run(preds, golds, family_filters, code_dict_reverse, True)
    filtered_result = result[result["Identity Code"] !="Slack"]
    return filtered_result
    
if __name__=="__main__":
    #data_read
    preds = pd.read_csv("sample_predictions.csv").drop(columns=["Unnamed: 0"])
    golds = pd.read_csv("sample_gold.csv").drop(columns=["Unnamed: 0"])
    
    #sample filter_setup
    codes = list(preds.columns)
    code_dict = dict(zip(codes, range(len(codes))))
    code_dict_r = dict({code_dict[code]:code for code in code_dict.keys()})
    all_codes_list = list(codes)
    all_codes_parent_list = ([code.split(".")[0] for code in all_codes_list])
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
    
    preds_converted = (preds>0.5)*1
    golds_converted = (golds>0.5)*1
    code_dict_r[99]="OOF"
    co_results = coocurrence(preds_converted, golds_converted, family_filters, code_dict_r)
    co_results.to_csv("pred_gold_coocurrence.csv", index=False)
    print("vanilla")
    hi_results = hierarchical_confusion_unscaled(preds_converted, golds_converted, family_filters, code_dict_r)
    hi_results.to_csv("pred_gold_hierarchical_confusion.csv", index=False)
    print("transposed")
    hi_results_T = hierarchical_confusion_unscaled(golds_converted, preds_converted, family_filters, code_dict_r)
    hi_results.to_csv("gold_pred_hierarchical_confusion.csv", index=False)