import pytest

import json
from unittest.mock import patch

from requests.models import Response

from configuration_optimizer import ConfigurationOptimizer


@pytest.fixture
def configuration_optimizer():
    return ConfigurationOptimizer(
        base_uri="localhost:1234",
        target="accuracy",
        max_iter=10
    )


def get_scores_mock(*args, **kwargs):
    response_data = {
        "labelled_scores": [
            [6.83771, 1],
            [3.37173, 0],
            [4.37317, 1],
            [0.59144, 0]
        ]
    }
    content = json.dumps(response_data)

    response = Response()
    response.status_code = 200
    response._content = str.encode(content)

    return response


@patch("requests.get", side_effect=get_scores_mock)
def test_get_scores(mock_get, configuration_optimizer):
    scores, labels = configuration_optimizer._scores()

    assert 6.83771 in scores
    assert labels[0] == 1
