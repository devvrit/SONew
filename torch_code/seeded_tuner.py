from nni.algorithms.hpo.hyperopt_tuner import json2space, HyperoptTuner
from nni.common.hpo_utils.validation import validate_search_space

import hyperopt as hp
import numpy as np

class SeededTuner(HyperoptTuner):

    def __init__(self, seed, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = int(seed)

    def update_search_space(self, search_space):
        validate_search_space(search_space)
        self.json = search_space

        search_space_instance = json2space(self.json)
        rstate = np.random.RandomState(self.seed)
        trials = hp.Trials()
        domain = hp.Domain(None,
                           search_space_instance,
                           pass_expr_memo_ctrl=None)
        algorithm = self._choose_tuner(self.algorithm_name)
        self.rval = hp.FMinIter(algorithm,
                                domain,
                                trials,
                                max_evals=-1,
                                rstate=rstate,
                                verbose=0)
        self.rval.catch_eval_exceptions = False