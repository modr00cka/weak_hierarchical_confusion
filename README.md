# Hierarchical Confusion Matrices
## Motivation

Confusion matrices are useful evaluation analysis tools in strongly-labelled scenarios. As long as we know exactly which prediction (or lack thereof) is associated with which gold label (or lack thereof), we can very easily count how many times each predicted class is associated with each gold standard class.

The simplest scenario, of course, is where each input is associated with 1 label, and the model produces a prediction for a single class - a single-label classification. In such a scenario we a predicted label either matches the gold standard label, or is mismatched (confused) with a different valid label. Thus a confusion matrix holds the counts of correct predictions (traditionally the entries on the diagonal), and counts of misclassifications between the predicted and expected class (non-diagonal entries). A high misclassification between two classes indicates that, with respect to the model's parameters, members of these classes are similar and difficult to distinguish.

A slightly more complex scenario is one where *n* predictions are to be matched with *m* gold standard labels, while there is some clear form of associating a predicted label with a gold standard label - e.g., assigning said prediction and gold standard label to the same span of text in a NER+L task. Apart from the two scenarios we described for the simplest scenario, one additional scenario arises, where for a prediction assigned to a particular span of text there is no corresponding gold standard label (or vice versa). Hence, apart from the count of correct predictions, and misclassifications, we can also track over/under predictions (e.g., through an extra invalid class representing absence of a class).

The confusion matrix is a powerful tool, enabling highl-evel error analysis beyond tracking precision/recall of the model.A confusion matrix indicates which classes tend to be confused with each other, or missed completely. Such error analysis can then be used in further model design, or be an integral part of a deployed model, where a user can approach the model's predictions with access to prior knowledge of the model's shortcomings. In any way, such an analysis done on the level of class with respect to the rest of the label set is more indicative of underlying issues than an aggregate measure of performance on individual labels in isolation.

In the weakly-labeled scenario, such as ICD-9 classification of MIMIC-III dicharge summaries (multi-label document classification) both predictions and gold labels are presented on the document level. Hence, unfortunately, we do not have access to links gold standard labels and text spans, and, consequently, with individual predictions. As such, if there is a mismatch between a predicted label and the gold standard, we cannot state with certainty that label A (e.g., alcoholism) was misclassified as label B (e.g., bronchitis), or whether the model underpredicted A, while overpredciting B. The simplest thing we can do is draw a co-occurrence matrix between predictions and gold labels - this can tell us which labels tend to be in the gold standard when a given label is in the prediction set, and vice versa. However, on its own, this data is not fine-grained enough for error analysis - in particular, the notion of error is not clear just from such co-occurence (particularly not when aggregating across multiple documents)

However, certain labels are more similar to each other within the label space. In the case of ICD-9, the labelspace itself is organised into a tree-structured ontology. The deeper the level of the ontology, the more specific a concept is, and the more similar it is to its sibling concepts. This is reflected in the verbal description of nodes within the ontology. There is no guarantee that a single predicted label is to be associated with a single gold standard label (1-to-1 correspondence). For example, assume labels A.1 and A.2 are siblings are hence very similar (e.g., hypertension unspecified, hypertension malignant), the model for some reason predicts both for a given data point, but only one of them, say A.2, appears in the gold standard - this is an example of overprediction within a family of (mutually-exclusive) codes. Would we in this situation consider A.1 to be mispredicted as A.2 (confusion), or have no gold label associated with it in the confusion matrix?

## Assumptions

If we take the problems of ontology-defined similarity and 1-to-1 correspondence, we define assumptions for a confusion-matrix-like analysis tool for the weakly-labelled hierarchical scenario:

Assumption (1) 1-to-1 true positive correspondence: If a label is present both in the prediction and gold-standard sets for a document, this is considered a True Positive, and is removed from mismatch calculation.

![1](Images/true_positive_assumption.png)
<br><br>
<br><br>
<center>Figure 1: 1-to-1 True Positive Correspondence Assumption. Codes 364.00, 364.02, 364.03, and 364.04 are predicted, while codes 364.00, 364.01, and 364.02 are expected. Left: Co-occurrence matrix between prediction and gold standard; Right: Same matrix after applying the assumption - codes that match between prediction and gold standard are not considered in mismatches./center>
<br><br>

Assumption (2) keeping it within the family: Disregarding true positives, non-true-positive codes in the prediction set are matched with non-true-postive codes in gold standard set, within the same family of codes. That is to say, if A.1, A.2, B.1 are predicted and A.1, A.3, B.2, are expected, there is a match A.1-A.1, the mismatches are A.2-A.3 (belonging to the A-family), and B.1-B.2 (belonging to the B-family).

![2](Images/within_family_assumption.png)
<br><br>
<br><br>
<center>Figure 2: Within-Family Confusoin Assumption. Codes 364.00, 364.02, 364.03, and 365.02 are predicted, while codes 364.00, 364.02, and 365.01 are expected. Left: Co-occurrence matrix between prediction and gold standard; Right: Same co-occurrence matrix with within-family confusion assumption. Mismatches are drawn only between codes within the same family</center>
<br><br>

Assumption (3) Out-Of-Family Scenario: If during mismatch calculation a code from one of the prediction/gold set cannot be match with any code from the other, as there is no code from that family present in the set, the code is associated with a special OOF code. For instance, if A.1, A.2, B.1 are predicted and A.1, A.3 are expected, A.1-A.1 are a match, A.2-A.3 are a mismatch, and as B.1 does not have a counterpart in the expected set, B.1 is associated with an OOF as mismatch (B.1-OOF). 

![3](Images/oof_assumption.png)
<br><br>
<br><br>
<center>Figure 3: The Out-of-Family Scenario. Codes 364.00 and 364.02 are predicted, while codes 364.00, 364.01, and 364.02 are expected. After applying Assumption (1) the code 364.01 present in the gold standard has no valid sibling code left in the prediction set to be mismatched with. Hence it is mismatched with the Out-of-Family label (OOF).</center>
<br><br>

## Use

For each family of codes we can derive a confusion matrix following these rules. That matrix can then be interpreted for different pruposes. We present a setup where the confusion matrices are used for assessing if a model predicts a certain code, if it is more likely to be the correct expected code, or if the gold standard is more likely to have a mismatched sibling code, or even an OOF. The same approach can be re-used (through transposed) to determine, given a gold standard label, which code from the same family is most likely to be predicted and whether that prediction would be a match.

