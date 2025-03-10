"""High level conversion functions."""

import numpy as np
import xarray as xr

try:
    from tree import is_nested
except ImportError:
    is_nested = lambda obj: False

from .base import dict_to_dataset
from .inference_data import InferenceData
from .io_beanmachine import from_beanmachine
from .io_cmdstan import from_cmdstan
from .io_cmdstanpy import from_cmdstanpy
from .io_emcee import from_emcee
from .io_numpyro import from_numpyro
from .io_pyro import from_pyro
from .io_pystan import from_pystan


def convert_from_stan_model(obj, group, **kwargs):
    """Convert a stan model object to an Inference Data object

    Paramters
    ---------
    obj : See convert_to_inference_data
    group : See convert_to_inference_data
    kwargs : See convert_to_inference_data

    Returns
    -------
    InferenceData
    """
    set_prior_or_posterior(group, **kwargs)
    if obj.__class__.__name__ == "CmdStanMCMC":
        return from_cmdstanpy(**kwargs)
    else:  # pystan or pystan3
        return from_pystan(**kwargs)


def set_prior_or_posterior(group, **kwargs):
    """Sets the prior or posterior value in kwargs based on the group value

    If group is "sample_stats", the posterior is set, if it is "sample_stats_prior", the prior is set

    Paramters
    ---------
    group : See convert_to_inference_data
    kwargs : See convert_to_inference_data
    """
    if group == "sample_stats":
        kwargs["posterior"] = kwargs.pop(group)
    elif group == "sample_stats_prior":
        kwargs["prior"] = kwargs.pop(group)


def convert_from_file(obj, group, **kwargs):
    """Convert file to Inference Data object

    Extracts the data from a csv or netcdf file to an inference data object

    Paramters
    ---------
    obj : string representing a path to a .csv or .nc file
    group : See convert_to_inference_data
    kwargs : See convert_to_inference_data
    """
    if obj.endswith(".csv"):
        set_prior_or_posterior(group, **kwargs)
        return from_cmdstan(**kwargs)
    else:
        if kwargs["coords"] is not None or kwargs["dims"] is not None:
            raise TypeError(
                "Cannot use coords or dims arguments reading InferenceData from netcdf."
            )
        return InferenceData.from_netcdf(obj)


def from_mcmc(obj, group, **kwargs):
    """Convert a MCMC object into an inference data object

    The exact method depends on the module of the object, if it is "pyro" or "numpyro"

    Parameters
    ----------
    obj : An MCMC object
    group : See convert_to_inference_data
    kwargs : See convert_to_inference_data
    """
    if obj.__class__.__module__.startswith("pyro"):
        return from_pyro(posterior=kwargs.pop(group), **kwargs)
    elif obj.__class__.__module__.startswith("numpyro"):
        return from_numpyro(posterior=kwargs.pop(group), **kwargs)


# pylint: disable=too-many-return-statements
def convert_to_inference_data(obj, *, group="posterior", coords=None, dims=None, **kwargs):
    r"""Convert a supported object to an InferenceData object.

    This function sends `obj` to the right conversion function. It is idempotent,
    in that it will return arviz.InferenceData objects unchanged.

    Parameters
    ----------
    obj : dict, str, np.ndarray, xr.Dataset, pystan fit
        A supported object to convert to InferenceData:
            | InferenceData: returns unchanged
            | str: Attempts to load the cmdstan csv or netcdf dataset from disk
            | pystan fit: Automatically extracts data
            | cmdstanpy fit: Automatically extracts data
            | cmdstan csv-list: Automatically extracts data
            | emcee sampler: Automatically extracts data
            | pyro MCMC: Automatically extracts data
            | beanmachine MonteCarloSamples: Automatically extracts data
            | xarray.Dataset: adds to InferenceData as only group
            | xarray.DataArray: creates an xarray dataset as the only group, gives the
                         array an arbitrary name, if name not set
            | dict: creates an xarray dataset as the only group
            | numpy array: creates an xarray dataset as the only group, gives the
                         array an arbitrary name
    group : str
        If `obj` is a dict or numpy array, assigns the resulting xarray
        dataset to this group. Default: "posterior".
    coords : dict[str, iterable]
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : dict[str, List(str)]
        A mapping from variables to a list of coordinate names for the variable
    kwargs
        Rest of the supported keyword arguments transferred to conversion function.

    Returns
    -------
    InferenceData
    """
    kwargs[group] = obj
    kwargs["coords"] = coords
    kwargs["dims"] = dims

    # Cases that convert to InferenceData
    if isinstance(obj, InferenceData):
        if coords is not None or dims is not None:
            raise TypeError("Cannot use coords or dims arguments with InferenceData value.")
        return obj
    elif isinstance(obj, str):
        return convert_from_file(obj, group, **kwargs)
    elif (
        obj.__class__.__name__ in {"StanFit4Model", "CmdStanMCMC"}
        or obj.__class__.__module__ == "stan.fit"
    ):
        return convert_from_stan_model(obj, group, **kwargs)
    elif obj.__class__.__name__ == "EnsembleSampler":  # ugly, but doesn't make emcee a requirement
        return from_emcee(sampler=kwargs.pop(group), **kwargs)
    elif obj.__class__.__name__ == "MonteCarloSamples":
        return from_beanmachine(sampler=kwargs.pop(group), **kwargs)
    elif obj.__class__.__name__ == "MCMC":
        return from_mcmc(obj, group, **kwargs)

    # Cases that convert to xarray
    if isinstance(obj, xr.Dataset):
        dataset = obj
    elif isinstance(obj, xr.DataArray):
        if obj.name is None:
            obj.name = "x"
        dataset = obj.to_dataset()
    elif isinstance(obj, dict):
        dataset = dict_to_dataset(obj, coords=coords, dims=dims)
    elif is_nested(obj) and not isinstance(obj, (list, tuple)):
        dataset = dict_to_dataset(obj, coords=coords, dims=dims)
    elif isinstance(obj, np.ndarray):
        dataset = dict_to_dataset({"x": obj}, coords=coords, dims=dims)
    elif isinstance(obj, (list, tuple)) and isinstance(obj[0], str) and obj[0].endswith(".csv"):
        set_prior_or_posterior(group, **kwargs)
        return from_cmdstan(**kwargs)
    else:
        allowable_types = (
            "xarray dataarray",
            "xarray dataset",
            "dict",
            "pytree (if 'dm-tree' is installed)",
            "netcdf filename",
            "numpy array",
            "pystan fit",
            "emcee fit",
            "pyro mcmc fit",
            "numpyro mcmc fit",
            "cmdstan fit csv filename",
            "cmdstanpy fit",
        )
        raise ValueError(
            f'Can only convert {", ".join(allowable_types)} to InferenceData, '
            f"not {obj.__class__.__name__}"
        )

    return InferenceData(**{group: dataset})


def convert_to_dataset(obj, *, group="posterior", coords=None, dims=None):
    """Convert a supported object to an xarray dataset.

    This function is idempotent, in that it will return xarray.Dataset functions
    unchanged. Raises `ValueError` if the desired group can not be extracted.

    Note this goes through a DataInference object. See `convert_to_inference_data`
    for more details. Raises ValueError if it can not work out the desired
    conversion.

    Parameters
    ----------
    obj : dict, str, np.ndarray, xr.Dataset, pystan fit
        A supported object to convert to InferenceData:

        - InferenceData: returns unchanged
        - str: Attempts to load the netcdf dataset from disk
        - pystan fit: Automatically extracts data
        - xarray.Dataset: adds to InferenceData as only group
        - xarray.DataArray: creates an xarray dataset as the only group, gives the
          array an arbitrary name, if name not set
        - dict: creates an xarray dataset as the only group
        - numpy array: creates an xarray dataset as the only group, gives the
          array an arbitrary name

    group : str
        If `obj` is a dict or numpy array, assigns the resulting xarray
        dataset to this group.
    coords : dict[str, iterable]
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : dict[str, List(str)]
        A mapping from variables to a list of coordinate names for the variable

    Returns
    -------
    xarray.Dataset
    """
    inference_data = convert_to_inference_data(obj, group=group, coords=coords, dims=dims)
    dataset = getattr(inference_data, group, None)
    if dataset is None:
        raise ValueError(
            "Can not extract {group} from {obj}! See {filename} for other "
            "conversion utilities.".format(group=group, obj=obj, filename=__file__)
        )
    return dataset
