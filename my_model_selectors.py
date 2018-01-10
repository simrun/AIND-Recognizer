import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

        self.features = len(self.sequences[0][0])
        self.datapoints = len(self.sequences[0])

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    where   L is the likelihood
            p is the number of free parameters in the HMM
                transition parameters: states - 1 (we never transition out of the last state)
                output parameters: 2 * states * features (each gaussian has a mean and a variance)
            N is the number of frames (in all sequences)
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        candidates = {np.inf: None}
        for n in range(self.min_n_components, self.max_n_components + 1):
            model = self.base_model(n)
            if model == None:
                continue

            try:
                logL = model.score(self.X, self.lengths)
            except:
                continue
            p = n - 1 + 2*n*self.features
            logN = np.log(self.datapoints)
            bic = -2*logL + p*logN

            candidates[bic] = model

        return candidates[min(candidates)]


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        other_words = self.hwords.copy()
        other_words.pop(self.this_word)

        candidates = {-np.inf: None}
        for n in range(self.min_n_components, self.max_n_components + 1):
            model = self.base_model(n)
            if model == None:
                continue

            try:
                thisL = model.score(self.X, self.lengths)
            except:
                continue

            otherL = [model.score(Y, lengths) for Y, lengths in other_words.values()]
            dic = thisL - sum(otherL)/len(otherL)

            candidates[dic] = model

        return candidates[max(candidates)]


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        kf = KFold()
        try:
            folds = list(kf.split(self.sequences))
        except:
            return None

        scores = {}
        for n in range(self.min_n_components, self.max_n_components + 1):
            fold_scores = []
            for train_idx, test_idx in folds:
                train_Xlengths = combine_sequences(train_idx, self.sequences)
                test_Xlengths = combine_sequences(test_idx, self.sequences)

                try:
                    model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(*train_Xlengths)
                    fold_scores.append(model.score(*test_Xlengths))
                except:
                    continue

            if not fold_scores:
                continue
            else:
                avg_fold_score = sum(fold_scores)/len(fold_scores)
                scores[avg_fold_score] = n

        try:
            n = scores[max(scores)]
            model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
        except:
            return None
        return model