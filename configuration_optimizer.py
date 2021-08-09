"""Optimize a black-box algorithm with constrained configuration parameters."""

from typing import Union, Dict, Tuple, List

import requests

from bayes_opt import BayesianOptimization


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
                 target: str,
                 max_iter: int,
                 seed: int = 0) -> None:
        """
        Parameters:
            base_uri: The base URI of the HTTP API.
            target: One of "f1" or "accuracy".
            max_iter: The maximum number of iterations that the optimizer will
                      perform.
            seed: The random seed of the algorithm.
        """
        self.base_uri = base_uri

        self.target = target
        self.max_iter = max_iter

        self.seed = seed

        # TODO: initial values
        self._current_config = None
        self._scores_cache = {}
        self._labels_cache = {}

    def start(self) -> None:
        """
        Execute the optimization procedure.
        """
        # Fetch the configuration specification
        config_spec = self._config_spec()

        # Build the optimizer
        self._optim = BayesianOptimization(
            f=self._configure_and_evaluate,
            pbounds=config_spec,
            random_state=self.seed
        )

        # Perform AUROC optimization
        self._optim.maximize(init_points=2, n_iter=self.max_iter)

        # Threshold optimization
        best_config = self._optim.max["params"]
        thresholds = self._optimize_thresholds(config=best_config)

        # Set final configuration
        self._configure(config, thresholds, True)

    def _configure_and_evaluate(self, **kwargs) -> float:
        """
        The main portion of each iteration. Configure the algorithm with a
        given set of parameters, and evaluate the parameters based on AUROC.

        Returns:
            auroc: The AUROC at the given set of parameters.
        """
        self._configure(kwargs)
        scores, labels = self._scores()
        auroc = roc_auc_score(labels, scores)

        # TODO cache results in case program dies

        return auroc

    def _config_spec(self) -> Dict[str, Tuple[float, float]]:
        """
        Fetch the configuration specification.

        Returns:
            parsed_config_spec: A dictionary mapping the name of each variable
                                to a 2-entry tuple, where the first element is
                                the lower bound of the parameter and the second
                                is the upper bound.
        """
        uri = self.base_uri + "/configSpec"
        req = requests.get(uri)
        # TODO error handle
        config_spec = req.json()["configuration_specification"]

        parsed_config_spec = {
            var_name: tuple(bounds) for var_name, bounds in config_spec.items()
        }

        return parsed_config_spec

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
        uri = self.base_uri + "/config"
        req = requests.post(
            uri,
            data={"config": config,
                  "thresholds": thresholds,
                  "optimal": optimal}
        )
        # TODO: error handle and raise if problem
        response = req.json()
        self._current_config = config

    def _scores(self) -> Tuple[List[float], List[int]]:
        """
        Fetch the scores of the classifier under the current configuration.

        Returns:
            scores: The list of classifier scores of length N.
            labels: The corresponding labels (0 or 1) of length N.
        """
        uri = self.base_uri + "/scores"
        req = requests.get(uri)
        response = req.json()
        # TODO error handle
        scores_labels = response["labelled_scores"]

        scores, labels = tuple(map(list, list(zip(*scores_labels))))

        self._scores_cache[self._current_config] = scores
        self._labels_cache[self._current_config] = labels

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
        scores = self._scores_cache[config]
        labels = self._labels_cache[config]

        # TODO optimize the target here

        thresholds = 2., 4.

        return thresholds


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--target", type=str, choices=["f1", "accuracy"], required=True)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--output_filename", type=str, required=True)
    parser.add_argument("--resume_filename", type=str, required=False)
    args = parser.parse_args()

    optimizer = ConfigurationOptimizer(
        hostname=args.hostname,
        port=args.port,
        target=args.target,
        max_iter=args.max_iter
    )
    if args.resume_filename is None:
        optimizer.start(output_filename=args.output_filename)
    else:
        optimizer.resume(resume_filename=args.resume_filename)
