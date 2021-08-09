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


def update_config_no_thresholds_mock(*args, **kwargs):
    post_data = kwargs["data"]
    assert post_data["config"] == {
        "name_m": 0.2, "name_u": 0.7
    }

    response_data = {
        "configuration_updated": True
    }
    content = json.dumps(response_data)

    response = Response()
    response.status_code = 200
    response._content = str.encode(content)

    return response


def update_config_with_thresholds_mock(*args, **kwargs):
    post_data = kwargs["data"]
    assert post_data["config"] == {
        "name_m": 0.2, "name_u": 0.7
    }
    assert post_data["thresholds"] == [2., 4.]
    assert post_data["optimal"]

    response_data = {
        "configuration_updated": True
    }
    content = json.dumps(response_data)

    response = Response()
    response.status_code = 200
    response._content = str.encode(content)

    return response




@patch("requests.post", side_effect=update_config_no_thresholds_mock)
def test_configure_with_no_thresholds(mock_get, configuration_optimizer):
    configuration_optimizer._configure(
        {"name_m": 0.2, "name_u": 0.7}
    )

@patch("requests.post", side_effect=update_config_with_thresholds_mock)
def test_configure_with_thresholds(mock_get, configuration_optimizer):
    configuration_optimizer._configure(
        {"name_m": 0.2, "name_u": 0.7},
        [2., 4.],
        True
    )
