# define processor for worldclim data
# define dataset for worldclim data

import xarray
import zarr
from numcodecs.zarr3 import PCodec
import numpy
import os
from torch.utils.data import Dataset, IterableDataset
from helpers import set_seed, set_device, get_logger
import warnings
from tqdm import tqdm
import h5py
import json


LOGGER = get_logger(name = "worldclim-dataset", log_file="worldclim-dataset.log")


class WorldClimProcessor():
    def __init__(self, path):
        super().__init__()
        self.path = path


    def _find_dataset(self, path: str) -> str:
        """
        Checks that the dataset exists at the given path

        Args:
            path (str): The path to the dataset

        Returns:
            str: Confirmation that the dataset exists

        Raises:
            FileNotFoundError: If the dataset does not exist
        """
        LOGGER.info(f"Checking that dataset exists at {path}")
        if os.path.exists(path):
            LOGGER.info(f"Dataset found at {path}")
            LOGGER.info("____________________________________________")
            return f"Dataset found at {path}"
        else:
            raise FileNotFoundError(f"Dataset not found at {path}")

    def _load_dataset(self,path: str) -> xarray.Dataset:
        """
        Loads the dataset from the given path

        Args:
            path (str): The path to the dataset

        Returns:
            xarray.Dataset: The dataset
        """
        LOGGER.info(f"Loading dataset from {path}")
        warnings.filterwarnings(
            "ignore",
            message="Numcodecs codecs are not in the Zarr version 3 specification.*",
            category=UserWarning,
            module="numcodecs.zarr3"
        )
        try:
            dataset = xarray.open_zarr(path)
            dataset = dataset
            print(f"Dataset loaded from {path}")
            LOGGER.info("DATASET LOADED")
            LOGGER.info("____________________________________________")
            return dataset
        except Exception as e:
            raise ValueError(f"Error loading dataset from {path}: {e}")

    def _land_compute_mask(self, dataset: xarray.Dataset, land_mask_value: float = -32768,
                           land_mask_variable: str = 'elev') -> xarray.DataArray:
        """
        Computes the mask for the dataset

        Args:
            dataset (xarray.Dataset): The dataset
            land_mask_value (float): The value to use for the mask
            land_mask_variable (str): The variable to use for the mask. Defaults to 'elev'.

        Returns:
            xarray.DataArray: The mask
        """
        LOGGER.info(f"Computing mask for {land_mask_variable} with value {land_mask_value}")

        # check that the variable exists
        if land_mask_variable not in dataset.data_vars:
            raise ValueError(f"Variable {land_mask_variable} not found in dataset")

        # create mask: True where the data is not equal to the mask_value
        land_mask = dataset[land_mask_variable].isel(t=0) != land_mask_value

        land_count = land_mask.values.sum()
        total_count = land_mask.values.size
        print(f"Land count: {land_count}")
        print(f"Total count: {total_count}")
        print(f"percentage land: {land_count / total_count * 100}")
        LOGGER.info(f"Land count: {land_count}")
        LOGGER.info(f"Total count: {total_count}")
        del land_count
        del total_count

        LOGGER.info(f"MASK COMPUTED FOR {land_mask_variable} WITH VALUE {land_mask_value}")
        print(f"Mask computed for {land_mask_variable} with value {land_mask_value}")
        LOGGER.info("____________________________________________")
        return land_mask

    def _split_land_ocean_coord_generator(self, mask: xarray.DataArray):
        """
        Splits the dataset into land and ocean coordinates

        Args:
            mask (xarray.DataArray): The mask for the dataset

        Returns:
            tuple: The land and ocean datasets
        """
        LOGGER.info("Splitting dataset into land and ocean based on mask")

        x_values = mask.coords['x'].values
        y_values = mask.coords['y'].values

        width = len(x_values)

        land_mask_values = mask.values.ravel()
        total = len(land_mask_values)

        for i, is_land in enumerate(tqdm(land_mask_values, desc="processing the land mask", total=total)):
            row = i // width
            col = i % width
            coord = (y_values[row], x_values[col])
            yield ('land' if is_land else 'ocean'), coord

    def _split_land_ocean_coords(self, land_mask: xarray.DataArray, output_file: str):
        """
        Splits the land_mask into land and ocean coordinates, writes them to HDF5.

        Args:
            land_mask (xarray.DataArray): A boolean mask (True = land, False = ocean).
            output_file (str): Output path for HDF5 file.
        """
        LOGGER.info("SPLITTING LAND AND OCEAN COORDINATES")

        with h5py.File(output_file, "w") as f:
            land_coords = f.create_dataset("land_coords", (0, 2), maxshape=(None, 2), dtype="f8", compression="gzip")
            ocean_coords = f.create_dataset("ocean_coords", (0, 2), maxshape=(None, 2), dtype="f8", compression="gzip")

            land_count = 0
            ocean_count = 0

            for coord_type, coord in self._split_land_ocean_coord_generator(land_mask):
                if coord_type == 'land':
                    land_coords.resize((land_count + 1, 2))
                    land_coords[land_count] = coord
                    land_count += 1
                else:
                    ocean_coords.resize((ocean_count + 1, 2))
                    ocean_coords[ocean_count] = coord
                    ocean_count += 1

            # Optional: store metadata
            f.attrs["land_count"] = land_count
            f.attrs["ocean_count"] = ocean_count
            f.attrs["total"] = land_count + ocean_count

            LOGGER.info(f"FINISHED SPLITTING LAND AND OCEAN COORDS")
            LOGGER.info("____________________________________________")

            # Example of checking contents
            print(f"Land count: {land_count}")
            print(f"Ocean count: {ocean_count}")
            print(f"Total: {land_count + ocean_count}")

    def _get_normalized_stats(self, dataset: xarray.Dataset):
        """
        Computes the normalized statistics for the dataset

        Args:
            dataset (xarray.Dataset): The dataset

        Returns:
            dict: The normalized statistics for the dataset
        """
        LOGGER.info("COMMENCING COMPUTING NORMALIZED STATS")

        normalized_stats = {}
        condition = dataset['elev'] != -32768.0

        for variable in dataset.data_vars:
            LOGGER.info(f"Computing normalized statistics for {variable}")
            print(f"Computing normalized statistics for {variable}")

            masked = dataset[variable].where(condition)
            min_value = masked.min().compute()
            max_value = masked.max().compute()

            normalized_stats[variable] = [min_value.item(), max_value.item()]
            del min_value, max_value

        with open('../normalized_stats.json', 'w') as f:
            json.dump(normalized_stats, f)

        LOGGER.info("COMPLETED COMPUTING NORMALIZED STATS")
        LOGGER.info("____________________________________________")
        return normalized_stats

    def run(self, path: str):
        """
        Runs the processor

        Args:
            path (str): The path to the dataset
        """
        LOGGER.info("RUNNING")
        self._find_dataset(path) # check that the dataset exists
        dataset = self._load_dataset(path) # load the dataset
        mask = self._land_compute_mask(dataset) # compute the mask
        self._split_land_ocean_coords(mask, 'coordinates.h5') # split the dataset
        self._get_normalized_stats(dataset) # get the normalized stats
        LOGGER.info("COMPLETED RUNNING")
        LOGGER.info("____________________________________________")