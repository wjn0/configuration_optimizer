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


def config_spec_endpoint_mock(*args, **kwargs):
    response_data = {
        "configuration_specification": {
            "name_m": [0.0, 1.0],
            "name_u": [0.0, 1.0],
            "address[PrimaryHome]_m": [0.0, 1.0],
            "address[PrimaryHome]_u": [0.0, 1.0]
        }
    }
    content = json.dumps(response_data)

    response = Response()
    response.status_code = 200
    response._content = str.encode(content)

    return response


@patch("requests.get", side_effect=config_spec_endpoint_mock)
def test_get_configuration_specification(mock_get, configuration_optimizer):
    config_spec = configuration_optimizer._config_spec()

    assert "name_m" in config_spec
    assert config_spec["name_m"] == (0., 1.)
