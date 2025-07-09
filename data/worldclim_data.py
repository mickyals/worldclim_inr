# define processor for worldclim data
# define dataset for worldclim data

import xarray
import zarr
from numcodecs.zarr3 import PCodec
import numpy
import os
import torch
from torch.utils.data import Dataset, IterableDataset
from helpers import set_seed, set_device, get_logger
import warnings
from tqdm import tqdm
import h5py
import json
from typing import Tuple


LOGGER = get_logger(name = "worldclim-dataset", log_file="worldclim-dataset.log")


class WorldClimProcessor():
    def __init__(self, config):
        """
        Initializes the processor with the path to the dataset

        Args:
            config (dict): A dictionary containing the following keys:
                - data (str): The path to the dataset
        """
        super().__init__()
        self.path = config['data']


    def _find_dataset(self, path: str) -> str:
        """
        Checks that the dataset exists at the given path.

        Args:
            path (str): The path to the dataset

        Returns:
            str: Confirmation that the dataset exists

        Raises:
            FileNotFoundError: If the dataset does not exist
        """
        LOGGER.info(f"Checking that dataset exists at {path}")
        # Check if the dataset exists at the given path
        if os.path.exists(path):
            # If the dataset exists, log a success message
            LOGGER.info(f"Dataset found at {path}")
            # Return a success message
            return f"Dataset found at {path}"
        else:
            # If the dataset does not exist, raise a FileNotFoundError
            raise FileNotFoundError(f"Dataset not found at {path}")

    def _load_dataset(self, path: str) -> xarray.Dataset:
        """
        Loads the dataset from the given path. The dataset is loaded using xarray.open_zarr,
        which returns a xarray.Dataset object. The warnings are filtered out to avoid UserWarning
        messages about the numcodecs codecs not being in the Zarr version 3 specification.

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
            # Load the dataset using xarray.open_zarr
            dataset = xarray.open_zarr(path)
            # Set the dataset attribute
            dataset = dataset
            # Log a success message
            print(f"Dataset loaded from {path}")
            LOGGER.info("DATASET LOADED")
            LOGGER.info("____________________________________________")
            # Return the dataset
            return dataset
        except Exception as e:
            # Raise a ValueError if there is an error loading the dataset
            raise ValueError(f"Error loading dataset from {path}: {e}")

    def _land_compute_mask(self, dataset: xarray.Dataset, land_mask_value: float = -32768,
                           land_mask_variable: str = 'elev') -> xarray.DataArray:
        """
        Computes the mask for the dataset.

        Args:
            dataset (xarray.Dataset): The dataset to compute the mask for.
            land_mask_value (float): The value to use for the mask.
            land_mask_variable (str): The variable to use for the mask. Defaults to 'elev'.

        Returns:
            xarray.DataArray: The computed mask as a boolean array.
        """
        LOGGER.info(f"Computing mask for {land_mask_variable} with value {land_mask_value}")

        # Check that the variable exists in the dataset
        if land_mask_variable not in dataset.data_vars:
            raise ValueError(f"Variable {land_mask_variable} not found in dataset")

        # Create mask: True where the data is not equal to the mask_value
        land_mask = dataset[land_mask_variable].isel(t=0) != land_mask_value

        # Calculate and log the land count and total count
        land_count = land_mask.values.sum()
        total_count = land_mask.values.size
        print(f"Land count: {land_count}")
        print(f"Total count: {total_count}")
        print(f"Percentage land: {land_count / total_count * 100}")
        LOGGER.info(f"Land count: {land_count}")
        LOGGER.info(f"Total count: {total_count}")

        # Log the completion of mask computation
        LOGGER.info(f"MASK COMPUTED FOR {land_mask_variable} WITH VALUE {land_mask_value}")
        print(f"Mask computed for {land_mask_variable} with value {land_mask_value}")
        LOGGER.info("____________________________________________")

        return land_mask

    def _split_land_ocean_coord_generator(self, mask: xarray.DataArray):
        """
        Splits the dataset into land and ocean coordinates

        This function takes a boolean mask (True = land, False = ocean) and
        yields a tuple for each coordinate in the mask. The first element of
        the tuple is a string indicating whether the coordinate is land or
        ocean. The second element of the tuple is the coordinate itself.

        Args:
            mask (xarray.DataArray): The mask for the dataset

        Yields:
            tuple: A tuple containing the type of coordinate (land or ocean) and
                the coordinate itself
        """
        LOGGER.info("Splitting dataset into land and ocean based on mask")

        # Get the x and y coordinates from the mask
        x_values = mask.coords['x'].values
        y_values = mask.coords['y'].values

        # Calculate the width of the mask
        width = len(x_values)

        # Get the values of the mask and calculate the total number of values
        land_mask_values = mask.values.ravel()
        total = len(land_mask_values)

        # Iterate over the mask and yield a tuple for each coordinate
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

        Notes:
            The generated HDF5 file will have two datasets: "land_coords" and "ocean_coords".
            Each dataset will have shape (N, 2) where N is the number of coordinates of the
            respective type. The first column of each dataset will contain the y coordinates
            and the second column will contain the x coordinates.
        """
        LOGGER.info("SPLITTING LAND AND OCEAN COORDINATES")

        with h5py.File(output_file, "w") as f:
            # Create the datasets
            land_coords = f.create_dataset("land_coords", (0, 2), maxshape=(None, 2), dtype="f8", compression="gzip")
            ocean_coords = f.create_dataset("ocean_coords", (0, 2), maxshape=(None, 2), dtype="f8", compression="gzip")

            # Initialize counters
            land_count = 0
            ocean_count = 0

            # Iterate over the coordinates
            for coord_type, coord in self._split_land_ocean_coord_generator(land_mask):
                if coord_type == 'land':
                    # Resize the land dataset and add the new coordinate
                    land_coords.resize((land_count + 1, 2))
                    land_coords[land_count] = coord
                    land_count += 1
                else:
                    # Resize the ocean dataset and add the new coordinate
                    ocean_coords.resize((ocean_count + 1, 2))
                    ocean_coords[ocean_count] = coord
                    ocean_count += 1

            # Store metadata
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
        Computes the normalized statistics for the dataset.

        This function iterates over all variables in the dataset and computes the minimum
        and maximum values for each variable, excluding any values that are equal to the
        mask value (-32768.0). The results are stored in a dictionary with the variable
        names as keys and the minimum and maximum values as values.

        Args:
            dataset (xarray.Dataset): The dataset

        Returns:
            dict: The normalized statistics for the dataset
        """
        LOGGER.info("COMMENCING COMPUTING NORMALIZED STATS")

        # Initialize the dictionary to store the results
        normalized_stats = {}

        # Iterate over all variables in the dataset
        for variable in dataset.data_vars:
            LOGGER.info(f"Computing normalized statistics for {variable}")
            print(f"Computing normalized statistics for {variable}")

            # Create a masked version of the variable
            masked = dataset[variable].where(dataset['elev'] != -32768.0)

            # Compute the minimum and maximum values of the masked variable
            min_value = masked.min().compute()
            max_value = masked.max().compute()

            # Store the results in the dictionary
            normalized_stats[variable] = [min_value.item(), max_value.item()]

            # Clean up memory
            del min_value, max_value

        # Write the results to a JSON file
        with open('../normalized_stats.json', 'w') as f:
            json.dump(normalized_stats, f)

        LOGGER.info("COMPLETED COMPUTING NORMALIZED STATS")
        LOGGER.info("____________________________________________")
        return normalized_stats

    def run(self):
        """
        Runs the WorldClimProcessor.

        This function checks that the dataset exists and loads it. It then computes the mask
        and splits the dataset into land and ocean coordinates. Finally, it computes the
        normalized statistics for the dataset.
        """
        LOGGER.info("RUNNING")
        self._find_dataset(self.path) # check that the dataset exists
        dataset = self._load_dataset(self.path)  # load the dataset

        # Compute the mask
        mask = self._land_compute_mask(dataset)
        # Split the dataset into land and ocean coordinates
        self._split_land_ocean_coords(mask, output_file='coordinates.h5')

        # Compute the normalized statistics for the dataset
        if not os.path.exists('normalized_stats.json'):
            self._get_normalized_stats(dataset)

        LOGGER.info("COMPLETED RUNNING")
        LOGGER.info("____________________________________________")


class WorldClimDataset(IterableDataset):
    def __init__(self, config):
        """
        Initializes the WorldClimDataset with specified paths and parameters.

        Args:
            config (dict): A dictionary containing the following keys:
                - data_path (str): The path to the data file.
                - points_path (str): The path to the points file.
                - normalised_stats_path (str): The path to the normalized stats JSON file.
                - key (str, optional): The key to access specific coordinates in the points file.
                    Defaults to 'land_coords'.
                - shuffle (bool, optional): Whether to shuffle the data points. Defaults to True.
                - deg2rad (bool, optional): Whether to convert coordinates to radians. Defaults to True.
        """
        super().__init__()

        # Store the provided arguments as instance attributes
        self.data_path = config['data_path']
        self.points_path = config['points_path']
        self.normalised_stats_path = config['normalised_stats_path']
        self.key = config.get('data_key', 'land_coords')
        self.shuffle = config.get('shuffle', True)

        # Load the dataset from the specified data path
        self.data = self._get_data()

        # Create a sorted list of variable names in the dataset
        self.var_list = sorted(list(self.data.data_vars))

        # Load the points from the points file
        self.points = self._get_points()

    def _get_data(self):
        """
        Loads the dataset from the specified data path

        Returns:
            xarray.Dataset: The loaded dataset
        """
        return xarray.open_dataset(self.data_path)

    def _get_points(self):
        """
        Loads the points from the points file, shuffles them if specified,
        and yields them one by one.

        Yields:
            numpy.ndarray: A coordinate point (2,)
        """

        LOGGER.info("LOADING POINTS")
        with h5py.File('coordinates.h5', 'r') as f:
            # Get the coordinates dataset and store the number of points
            coords = f[self.key]
            self.num_points = coords.shape[0]

            # Create an array of indices for the points
            indices = numpy.arange(self.num_points)

            # Shuffle the indices if specified
            if self.shuffle:
                numpy.random.shuffle(indices)

            # Yield each coordinate point
            for idx in indices:
                yield coords[idx]



    def _get_normalised_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reads the normalised stats from a JSON file and returns them as tensors.

        The JSON file should contain a dictionary with variable names as keys
        and a list of two values as values. The first value is the minimum value
        for that variable, and the second value is the maximum value for that
        variable.

        The function returns two tensors: min_tensor and max_tensor. Both tensors
        have shape (1, n), where n is the number of variables in the dataset.

        min_tensor contains the minimum values for each variable, and max_tensor
        contains the maximum values for each variable.

        Returns:
            min_tensor (torch.Tensor): shape (1, n)
            max_tensor (torch.Tensor): shape (1, n)
        """

        LOGGER.info("LOADING NORMALIZED STATS")
        with open(self.normalised_stats_path, 'r') as f:
            stats = json.load(f)

        # Ensure consistent order of variables
        min_values = []
        max_values = []
        for k in sorted(stats.keys()):
            min_values.append(min(stats[k])) # shape (n,)
            max_values.append(max(stats[k])) # shape (n,)

        # Create tensors from the min and max values
        min_tensor = torch.tensor(min_values).unsqueeze(1) # shape (1, n)
        max_tensor = torch.tensor(max_values).unsqueeze(1) # shape (1, n)

        return min_tensor, max_tensor



    def _normalize_data(self, raw_data_tensor: torch.Tensor, min_tensor: torch.Tensor, max_tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the raw data tensor to the range [-1, 1] using the min and max tensors as the normalization factors.

        Args:
            raw_data_tensor (torch.Tensor): The tensor of raw data to be normalized. Shape: (n, 1)
            min_tensor (torch.Tensor): The tensor of minimum values for each variable. Shape: (1, n)
            max_tensor (torch.Tensor): The tensor of maximum values for each variable. Shape: (1, n)

        Returns:
            normalized_data_tensor (torch.Tensor): The normalized tensor. Shape: (n, 1)
        """
        LOGGER.info("NORMALIZING DATA")
        return 2 * (raw_data_tensor - min_tensor) / (max_tensor - min_tensor) - 1
  # shape (2, 1)


    def __len__(self):
        """
        Returns the number of coordinate points available in the dataset.

        Returns:
            int: The number of points.
        """
        return self.num_points


    def __iter__(self):
        """
        Iterates over the points in the dataset, normalizes the data at each point,
        and yields a dictionary containing the input and output tensors.

        The input tensor is either the coordinates in radians or the coordinates as is.
        The output tensor is the normalized data at the point.

        Returns:
            generator: A generator that yields a dictionary containing the input and output tensors.
        """

        min_tensor, max_tensor = self._get_normalised_stats()
        # Iterate over the points in the dataset and yield the input and output tensors
        for y, x in self._get_points():
            sample = {}

            # Get the raw data at the point
            raw_data = self.data[self.var_list].sel(x=x, y=y, method='nearest').to_array().values

            # Convert the raw data to a PyTorch tensor
            raw_data = torch.tensor(raw_data)

            # Normalize the data using the minimum and maximum values
            normalized_data = self._normalize_data(raw_data, min_tensor, max_tensor)

            # Get the coordinates as is
            input_tensor = torch.tensor([[x,y]]) # shape (2, 1)

            # Yield the input and output tensors
            yield {'input': input_tensor, 'target': normalized_data}



