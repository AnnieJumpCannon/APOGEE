
"""
APOGEE DR14 processing script for The Cannon.
"""

__author__ = [
    "Andy Casey <andy.casey@gmail.com>",
    "Gail Zasowski <gail.zasowski@gmail.com>"
]

import cPickle as pickle
import logging
import multiprocessing as mp
import numpy as np
import os
from glob import glob

import continuum


# Enable logging.
logger = logging.getLogger("apogee.dr14.tc")
logger.setLevel(logging.INFO) 

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)-8s] %(message)s"))
logger.addHandler(handler)


#------------------------------------------------------#
# 0. Download all the data and set required variables. #
#------------------------------------------------------#
DEBUG = True
THREADS = 8
PICKLE_PROTOCOL = -1
ORIGINAL_DATA_DIR = "/media/My Book/surveys/apogee/dr14/spectra/"
CANNON_DATA_DIR = "/media/My Book/surveys/apogee/dr14/tc-normalized-spectra/"

# The TRAINING_SET_CATALOG_PATH must be a table containing:
#   - telescope
#   - location_id
#   - apogee_id
# so that the filename can be reconstructed, and:
#   - the training set labels (whatever you want to train on).
TRAINING_SET_CATALOG_PATH = "dr14-training-set.fits"


#------------------------------------------------------#
# 1. Normalize and stack the individual visit spectra. #
#------------------------------------------------------#

clobber_normalization = False
normalization_kwds = {
    "regions": [
        (15140, 15812),
        (15857, 16437),
        (16472, 16960),
    ],
    "conservatism": (2.0, 0.1),
    "normalized_ivar_floor": 1e-4,
    "continuum_pixels": np.loadtxt("continuum.list", dtype=int)
}

individual_visit_paths = glob("{}/apo*/*/apStar*.fits".format(ORIGINAL_DATA_DIR))
N = len(individual_visit_paths)

# Create a function so that we can process the normalization in parallel.
def _process_normalization(input_path):
    """
    Produce pseudo-continuum-normalized data products for APOGEE DR14 apStar
    spectra.

    :param input_path:
        The local path of an apStar spectrum.

    :returns:
        A two-length tuple indicating: (1) whether the normalization was
        successful, and (2) the input_path provided. If `None` is provided in
        (1), it is because the output file already exists and we were not
        instructed to clobber it.
    """

    # Generate the output path for The Cannon-normalized spctra.
    apogee_id = input_path.split("-r7-")[-1][:-5]
    telescope, location_id = input_path.split("/")[-3:-1]

    output_path = os.path.join(
        CANNON_DATA_DIR, telescope, location_id, "{}.pkl".format(apogee_id))

    # Create the folder structure if it doesn't exist already.
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    # Check if this output file already exists.
    if os.path.exists(output_path) and not clobber_normalization:
        logger.info("Skipping normalization of {}..".format(input_path))
        return (None, input_path)

    try:
        stacked, visits, metadata = continuum.normalize_individual_visits(
            input_path, full_output=True, **normalization_kwds)

    except:
        logger.exception("Normalization failed on {}".format(input_path))
        return (False, input_path)

    metadata.update(APOGEE_ID=apogee_id)

    stacked = np.vstack(stacked)
    visits = np.vstack(visits)
    
    with open(output_path, "wb") as fp:
        pickle.dump(stacked, fp, PICKLE_PROTOCOL)

    with open("{}.visits".format(output_path), "wb") as fp:
        pickle.dump(visits, fp, PICKLE_PROTOCOL)

    with open("{}.meta".format(output_path), "wb") as fp:
        pickle.dump(metadata, fp, PICKLE_PROTOCOL)

    logger.info("Normalized spectra in {} successfully".format(input_path))

    return (True, input_path)

# Process the normalization in parallel.
logger.info("Initializing {} threads to perform pseudo-continuum-normalization"\
    .format(THREADS))
pool = mp.Pool(THREADS)
result = pool.map_async(_process_normalization, individual_visit_paths).get()
pool.close()
pool.join()

# TODO Do something with the failures in `result`?



#-------------------------------------------------------#
# 2. Construct a training set from the stacked spectra. #
#-------------------------------------------------------#



#-------------------#
# 3. Train a model. #
#-------------------#




# 4. Train a model.

# 5. Test a model on the individual visit spectra (stacked and unstacked).

# 6. Collect test results into a single file.

# 7. Make arrangements to accept Nobel Prize in Stockholm.
