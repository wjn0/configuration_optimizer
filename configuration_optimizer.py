"""Optimize a black-box algorithm with constrained configuration parameters."""

from typing import Union, Dict, Tuple, List

import json

import requests

from bayes_opt import BayesianOptimization, Events, JSONLogger, util

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

import numpy as np


class APIException(Exception):
    pass


class ConfigurationOptimizer:
    """
    Given an HTTP API that supports the following operations:

        * Get configuration specification
        * Update configuration
        * Extract classifier scores

    this class will optimize the configuration parameters of the classifier to:

        * Optimize the AUROC of the scores
        * Optimize the threshold(s) of the algorithm (according to accuracy or
          F1 score)

    To see the specific format that the HTTP API must support, see the unit
    tests.
    """

    def __init__(self,
                 base_uri: str,
                 configuration_id: str,
                 timeout: int,
                 num_random_points: int,
                 max_iter: int,
                 target: str,
                 max_unclassified: float,
                 verify: Union[bool, str] = True,
                 http_basic_username: str = None,
                 http_basic_password: str = None,
                 seed: int = 0) -> None:
        """
        Parameters:
            base_uri: The base URI of the HTTP API.
            configuration_id: The ID of the configuration which will be
                              optimized.
            timeout: The HTTP request timeout in the number of seconds.
            num_random_points: The number of random configurations to try prior
                               to using model-guided optimization.
            max_iter: The maximum number of iterations that the optimizer will
                      perform.
            target: One of "f1" or "accuracy".
            max_unclassified: The maximum proportion of unclassified data
                              allowed (between 0 and 1).
            verify: The path to a certificate file or a boolean as to whether
                    or not the SSL connection should be verified.
            http_basic_username: The username if using HTTP basic auth.
            http_basic_password: The password if using HTTP basic auth.
            seed: The random seed of the algorithm.
        """
        self.base_uri = base_uri
        self.verify = verify
        if http_basic_username is None:
            self.auth = None
        else:
            self.auth = (http_basic_username, http_basic_password)
        self.configuration_id = configuration_id
        self.timeout = timeout

        self.target = target
        self.num_random_points = num_random_points
        self.max_iter = max_iter
        self.max_unclassified = max_unclassified

        self.seed = seed

        self._current_config = None
        self._scores_cache = {}
        self._labels_cache = {}

    def start(self,
              resume_filename: str = None,
              output_filename: str = None) -> None:
        """
        Execute the optimization procedure.

        Args:
            resume_filename: The path to the logfile (JSON) to resume from.
            output_filename: The path to the logfile (JSON) to write to.
        """
        # Fetch the configuration specification
        config_spec = self._fetch_config_spec()

        # Specify variable order so that we can cache
        self._variables = list(config_spec.keys())

        # Build the optimizer and resume from data if given
        self._optim = BayesianOptimization(
            f=self._configure_and_evaluate,
            pbounds=config_spec,
            random_state=self.seed
        )
        if resume_filename is not None:
            util.load_logs(self._optim, logs=[resume_filename])

        # Set up output file is given
        if output_filename is not None:
            logger = JSONLogger(path=output_filename)
            self._optim.subscribe(Events.OPTIMIZATION_STEP, logger)

        # Fetch the current config and the corresponding scores and inform the
        # model
        self._current_config = self._fetch_current_config()
        auroc = self._evaluate()
        self._optim.register(
            params=self._current_config,
            target=auroc
        )

        # Perform AUROC optimization
        self._optim.maximize(init_points=self.num_random_points,
                             n_iter=self.max_iter)

        # Threshold optimization
        best_config = self._optim.max["params"]
        thresholds = self._optimize_thresholds(config=best_config)

        # Set final configuration
        self._configure(best_config, thresholds, True)

    def _configure_and_evaluate(self, **kwargs) -> float:
        """
        The main portion of each iteration. Configure the algorithm with a
        given set of parameters, and evaluate the parameters based on AUROC.

        Returns:
            auroc: The AUROC at the given set of parameters.
        """
        self._configure(kwargs)

        return self._evaluate()

    def _evaluate(self) -> float:
        print("Evaluating...")
        scores, labels = self._scores()
        auroc = roc_auc_score(labels, scores)
        print(f"  AUROC: {auroc}")

        return auroc

    def _fetch_config_spec(self) -> Dict[str, Tuple[float, float]]:
        """
        Fetch the configuration specification.

        Returns:
            parsed_config_spec: A dictionary mapping the name of each variable
                                to a 2-entry tuple, where the first element is
                                the lower bound of the parameter and the second
                                is the upper bound.
        """
        response = self.__http(f"/matchConfig/{self.configuration_id}/spec")

        parsed_config_spec = {
            json.dumps((r["key"], var_name)): r["bounds"]
            for r in response["attributes"]
            for var_name in r
            if var_name != "key" and var_name != "bounds"
        }

        return parsed_config_spec

    def _fetch_current_config(self) -> Dict[str, float]:
        """
        Fetch the current configuration.

        Returns:
            parsed_config: A dictionary mapping the name of each variable
                           to its current setting.
        """
        print("Fetching current configuration...")
        response = self.__http(f"/matchConfig/{self.configuration_id}")
        parsed_config = {
            json.dumps((r["key"], var_name)): r[var_name]
            for r in response["attributes"]
            for var_name in r
            if var_name != "key" and var_name != "bounds"
        }
        self._thresholds = [response["nonMatchThreshold"],
                            response["matchThreshold"]]

        return parsed_config

    def _configure(self,
                   config: Dict[str, float],
                   thresholds: Union[float, Tuple[float, float]] = tuple(),
                   optimal: bool = False) -> None:
        """
        Configure the classification algorithm.

        Parameters:
            config: The configuration for the scoring algorithm. A dictionary
                    mapping the name of each variable to its setting as a
                    float.
            thresholds: Optionally, the threshold(s) which produce the final
                        predictions. If a single float, scores will be
                        binarized about that value. If a tuple of two floats,
                        the first should be smaller than the second. In this
                        case, scores less than the first float will be
                        predicted 0, scores greater than the second will be
                        predicted 1, and those in-between will not have a final
                        prediction.
            optimal: A boolean indicating whether this is the final, optimal
                     configuration setting.
        """
        print("Configuring...")
        deserialized_config = {}
        for serialized_key, value in config.items():
            vargroup, varname = json.loads(serialized_key)
            if vargroup not in deserialized_config:
                deserialized_config[vargroup] = {}
            deserialized_config[vargroup][varname] = value
        parsed_config = [
            {"key": vargroup_name,
             **{varname: value for varname, value in vargroup.items()}}
            for vargroup_name, vargroup in deserialized_config.items()
        ]
        req_data = {"attributes": parsed_config}
        if not thresholds:
            thresholds = self._thresholds
        req_data["nonMatchThreshold"] = thresholds[0]
        req_data["matchThreshold"] = thresholds[1]

        response = self.__http(
            f"/matchConfig/{self.configuration_id}",
            data=req_data
        )

        # TODO check response?
        # if not response["configuration_updated"]:
        #     raise APIException("Configuration was not updated")

        self._current_config = config

    def _scores(self) -> Tuple[List[float], List[int]]:
        """
        Fetch the scores of the classifier under the current configuration.

        Returns:
            scores: The list of classifier scores of length N.
            labels: The corresponding labels (0 or 1) of length N.
        """
        print("  Fetching scores...")
        scores_labels = self.__http(
            f"/matchConfig/{self.configuration_id}/$groundTruthScores"
        )

        scores, labels = [], []
        for label, label_scores in scores_labels.items():
            scores += label_scores
            labels += [int(label)] * len(label_scores)

        _cur_config = tuple(self._current_config[varname]
                            for varname in self._variables)
        self._scores_cache[_cur_config] = scores
        self._labels_cache[_cur_config] = labels

        return scores, labels

    def _optimize_thresholds(
        self,
        config: Dict[str, float]
    ) -> Union[float, Tuple[float, float]]:
        """
        Optimize the thresholds of the final classifier.
        
        Parameters:
            config: The optimal configuration, used to fetch the scores and
                    labels from the cache.

        Returns:
            threhsolds: The threshold(s) for the classifier.
        """
        print("Optimizing thresholds...")
        _config = tuple(config[varname] for varname in self._variables)
        scores = np.asarray(self._scores_cache[_config])
        labels = np.asarray(self._labels_cache[_config]).astype(int)

        # For a lower threshold based on score quantiles, compute the upper
        # threshold such that at most `self.max_unclassified` data is
        # unclassified and compute the corresponding f1 or accuracy score
        metric = {}
        for lower_q in np.arange(0., 1., step=0.001):
            upper_q = lower_q + self.max_unclassified
            pred = np.asarray([-9] * len(labels)).astype(int)
            if upper_q >= 1.:
                break
            lower_thresh = np.quantile(scores, lower_q)
            upper_thresh = np.quantile(scores, upper_q)
            pred[scores < lower_thresh] = 0
            pred[scores > upper_thresh] = 1
            sel = (pred >= 0)
            if self.target == "f1":
                metric_fn = f1_score
            elif self.target == "accuracy":
                metric_fn = accuracy_score
            metric[(lower_thresh, upper_thresh)] = metric_fn(labels[sel],
                                                             pred[sel])
        thresholds = max(metric.keys(), key=lambda k: metric[k])

        print(f"Achieved {self.target} of {metric[thresholds]} "
              f"with thresholds {thresholds}")

        return thresholds

    def __http(self,
               rel_path: str,
               data: Dict = None) -> Dict:
        uri = self.base_uri + rel_path 
        if data is None:
            req = requests.get(
                uri, timeout=self.timeout, auth=self.auth
            )
        else:
            req = requests.put(
                uri, json=data, timeout=self.timeout, auth=self.auth
            )

        if req.status_code != 200:
            try:
                exc = req.json()
                if "message" in exc:
                    message = exc["message"]
                else:
                    message = f"JSON parsed, but no message element: {exc}"
            except:
                message = "No valid JSON found"

            raise APIException(f"Status code {req.status_code}. {message}")

        response = req.json()

        return response


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_uri", type=str, required=True)
    parser.add_argument("--ssl_cert", type=str, default=False)
    parser.add_argument("--http_basic_username", type=str, default=None)
    parser.add_argument("--http_basic_password", type=str, default=None)
    parser.add_argument("--configuration_id", type=str, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--num_random_points", type=int, default=2)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--target", type=str, choices=["f1", "accuracy"], required=True)
    parser.add_argument("--max_unclassified", type=float, default=0.2)
    parser.add_argument("--output_filename", type=str, required=True)
    parser.add_argument("--resume_filename", type=str, required=False)
    args = parser.parse_args()

    optimizer = ConfigurationOptimizer(
        base_uri=args.base_uri,
        verify=args.ssl_cert,
        configuration_id=args.configuration_id,
        timeout=args.timeout,
        num_random_points=args.num_random_points,
        max_iter=args.max_iter,
        target=args.target,
        max_unclassified=args.max_unclassified,
        http_basic_username=args.http_basic_username,
        http_basic_password=args.http_basic_password
    )
    optimizer.start(resume_filename=args.resume_filename,
                    output_filename=args.output_filename)
