'''
Trains the classifier with respective data.
'''

globalSequences = {}

priorCount = {
    "positive" : {},
    "negative" : {}
}

priors = {
    "positive" : 0.0,
    "negative" : 0.0
}
