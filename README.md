# SingnalPeptide-prediction
Predicting signal peptides in proteomes.

Run using:
classifier (prefix of experiment file) (method) (residues in sequence to be trained)

Example: 
classifier gallus svm 100 

This would attempt to predict the signal peptides in the gallus_positive and gallus_negative file using the svm algorithm, on a classifier trained on the first 100 amino acids in a sequence. 
