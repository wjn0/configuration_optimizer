# Black box classifier optimization over HTTP

Given an HTTP API which provides access to a classifier, optimize performance
metrics.

## Set up

    $ git clone https://github.com/hamilton-health-sciences/configuration_optimizer.git
    $ cd configuration_optimizer
    $ python3 -m pip install -r requirements.py

## Usage

Launch ML tool, pointing it at the dummy server:

    $ python3 -m configuration_optimizer --base_uri http://middleware-server:5000 --target f1 --max_iter 50 --output_filename logs.json

Full options:

    $ python3 -m configuration_optimizer -h
