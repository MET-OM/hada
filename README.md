# HADA

## Setup

```
$ mamba env create -f environment.yml
$ conda activate hada
$ pip install -e .
$ hada --help
```

## Usage

Check out `hada --help`.

Retrieve all configured variables for the last 24 hours:

```
$ conda activate hada
$ hada --output last_24h.nc
```

Increase debug output, configure the bounding box (in longitude, latitude, resolution)
and select only variables containing 'temp':

```
$ hada --log-level=debug -v temp --bbox-deg 5,6,64,64.3 --dx 0.01 --dy 0.01 --output test3.nc
```

## Installing without conda

The description under "Setup" above is based on running `hada` in a `conda` environment.
You can also install `hada` in a standard Python virtual environment.

1. Install a Python interpreter (e.g. version 3.11.x)
1. Install [Poetry](https://python-poetry.org/docs/#installation) (>= version 1.2)

Create a virtual environment with `hada` installed:

```
$ poetry install --without dev
```

Tip: If you're using a Debian-based distribution, you might need to:

1. Install the `libgeos-dev` in order to build/install the `cartopy` package:
   ```
   sudo apt-get install libgeos-dev
   ```
1. Make the `ca-certificates` available under an alternative location:
   ```
   sudo mkdir -p /etc/pki/tls/certs
   sudo ln -s /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt
   ```
