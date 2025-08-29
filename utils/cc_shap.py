import numpy as np
from scipy import spatial, stats, special
from sklearn import metrics
from difflib import SequenceMatcher
from utils import setup_logger

logger = setup_logger("experiment_logger")

def explain_lm(s, explainer, model, max_new_tokens=100, target = None):
    """ Compute Shapley Values for a certain model and tokenizer initialized in explainer. """
    model.generation_config.max_new_tokens = max_new_tokens
    model.config.max_new_tokens = max_new_tokens
    model.generation_config.do_sample = False
    model.generation_config.temperature = 1.0 
    
    if target is None:
        shap_vals = explainer([s])
    else:
        shap_vals = explainer([s], [target])

    return shap_vals

def aggregate_values_explanation(shap_values, tokenizer, roi=''):
    full_tokens = [t.strip() for t in shap_values.data[0].tolist()]
    roi_tokens = tokenizer.convert_ids_to_tokens(tokenizer(roi, padding=False, add_special_tokens=False).input_ids)
    roi_tokens = [t.replace('▁', ' ').lstrip('Ġ').strip() for t in roi_tokens]
    
    s = SequenceMatcher(a=full_tokens, b=roi_tokens)
    match = s.find_longest_match(alo=0, ahi=len(full_tokens), blo=0, bhi=len(roi_tokens))
    roi_start, roi_end = match.a, match.a + match.size

    add_to_base = np.abs(np.hstack((shap_values.values[:, :roi_start], shap_values.values[:, roi_end:]))).sum(axis=1)
    # check if values per output token are not very low as this might be a problem because they will be rendered large by normalization.
    logger.debug(f"small val check: {np.concatenate([shap_values.values[0, :roi_start], shap_values.values[0, roi_end:]])}")
    small_values = [True if x < 0.01 else False for x in np.mean(np.abs(np.concatenate([shap_values.values[0, :roi_start], shap_values.values[0, roi_end:]])), axis=0)]
    if any(small_values):
        logger.warning("Warning: Some output expl. tokens have very low values. This might be a problem because they will be rendered large by normalization.")
    logger.debug(f"roi tokens: {roi_tokens}\ninput tokens full: {shap_values.data[0].tolist()}\ninput tokens clipped: {shap_values.data[0].tolist()[roi_start:roi_end]}")
    ratios = shap_values.values / (np.abs(shap_values.values).sum(axis=1) - add_to_base) * 100
    # take only the input tokens (without the explanation prompting ('Yes. Why?'))
    return np.mean(ratios, axis=2)[0, roi_start:roi_end] #, len_to_marginalize # we only have one explanation example in the batch

def cc_shap_score(ratios_prediction, ratios_explanation):
    if np.isnan(ratios_prediction).any() or np.isnan(ratios_explanation).any():
        print("One of the arrays contains NaN values.")
        print(f"ratios prediction: {ratios_prediction}")
        print(f"ratios explanation: {ratios_explanation}")
    
        # create masks to ignore NaNs
        mask = ~np.isnan(ratios_prediction) & ~np.isnan(ratios_explanation)
    
        # apply the mask to both arrays
        ratios_prediction = ratios_prediction[mask]
        ratios_explanation = ratios_explanation[mask]
    
        # compute the cosine distance
        if len(ratios_prediction) == 0 or len(ratios_explanation) == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan # if there are no overlapping non-NaN values, return NaN
    
    cosine = spatial.distance.cosine(ratios_prediction, ratios_explanation)
    mse = metrics.mean_squared_error(ratios_prediction, ratios_explanation)
    var = np.sum(((ratios_prediction - ratios_explanation)**2 - mse)**2) / ratios_prediction.shape[0]
    
    # how many bits does one need to encode P using a code optimised for Q. In other words, encoding the explanation from the answer
    kl_div = stats.entropy(special.softmax(ratios_explanation), special.softmax(ratios_prediction))
    js_div = spatial.distance.jensenshannon(special.softmax(ratios_prediction), special.softmax(ratios_explanation))

    spearman_r = stats.spearmanr(ratios_prediction, ratios_explanation)

    return cosine, mse, var, kl_div, js_div, spearman_r

def compute_cc_shap(model, tokenizer, values_prediction, values_explanation, roi='', plot=None, visualize=False):
    ratios_prediction = aggregate_values_explanation(values_prediction, tokenizer, roi=roi)
    ratios_explanation = aggregate_values_explanation(values_explanation, tokenizer, roi=roi)
    cosine, mse, var, kl_div, js_div, spearman_r = cc_shap_score(ratios_prediction, ratios_explanation)
    
    if plot == 'display' or visualize:
        print(f"The faithfulness score (cosine distance) is: {cosine:.3f}")
        print(f"The faithfulness score (MSE) is: {mse:.3f}")
        print(f"The faithfulness score (var) is: {var:.3f}")
        print(f"The faithfulness score (KL div) is: {kl_div:.3f}")
        print(f"The faithfulness score (JS div) is: {js_div:.3f}")
        print(f"The faithfulness score (spearmanr) is: {spearman_r}");
    return cosine