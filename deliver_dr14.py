
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
THREADS = 8
PICKLE_PROTOCOL = -1
ORIGINAL_DATA_DIR = "/media/My Book/surveys/apogee/dr14/spectra/"
CANNON_DATA_DIR = "/media/My Book/surveys/apogee/dr14/tc-normalized-spectra/"
CANNON_MODEL_DIR = "/media/My Book/surveys/apogee/dr14/models/"

# The labelled_set_path must be a table containing:
#   - telescope
#   - location_id
#   - apogee_id
# so that the filename can be reconstructed, and:
#   - the training set labels (whatever you want to train on).
LABELLED_SET_PATH = "apogee-dr14-giants-training-set.fits"
MODEL_NAME = "apogee-dr14-giants"

# The label names to use in the model.
LABEL_NAMES = ("TEFF", "LOGG", "FEH", "MG_H", "AL_H")
MODEL_ORDER = 2
MODEL_REGULARIZATION = 0


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
        A three-length tuple indicating: (1) whether the normalization was
        successful, (2) the `input_path`, and (3) the `output_path` if the
        normalization was successful. If `None` is provided in (1), it is 
        because the output file already exists and we were not instructed 
        to clobber it.
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
        return (None, input_path, output_path)

    try:
        stacked, visits, metadata = continuum.normalize_individual_visits(
            input_path, full_output=True, **normalization_kwds)

    except:
        logger.exception("Normalization failed on {}".format(input_path))
        return (False, input_path, None)

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

    return (True, input_path, output_path)

# Process the normalization in parallel.
logger.info(
    "Starting {} threads to perform pseudo-continuum-normalization of {} stars"\
    .format(THREADS, N))

pool = mp.Pool(THREADS)
normalized_result = \
    pool.map_async(_process_normalization, individual_visit_paths).get()
pool.close()
pool.join()

# If there were input spectra that failed, then show a summary.
_failed = []
for result, input_path, output_path in normalized_result:
    if result not in (True, None):
        _failed.append(input_path)

if _failed:
    logger.info("Summary of failures ({}):".format(len(_failed)))
    for input_path in _failed:
        logger.info("\t{}".format(input_path))

# Save the dispersion from one spectrum.
for result, input_path, output_path in normalized_result:
    if result in (True, None):
        
        with fits.open(input_path) as image:
            dispersion = \
                10**(image[0].header["CRVAL1"] + np.arange(image[0].data.size) \
                        * image[0].data.header["CDELT1"])
        
        with open(os.path.join(CANNON_DATA_DIR, "dispersion.pkl"), "wb") as fp:
            pickle.dump(dispersion, fp, PICKLE_PROTOCOL)

        break

#-------------------------------------------------------#
# 2. Construct a training set from the stacked spectra. #
#-------------------------------------------------------#
clobber_model = True
labelled_set = Table.read(LABELLED_SET_PATH)
N = len(labelled_set)

with open(os.path.join(CANNON_DATA_DIR, "dispersion.pkl"), "rb") as fp:
    dispersion = pickle.load(fp)
P = dispersion.size

normalized_flux = np.zeros((N, P), dtype=float)
normalized_ivar = np.zeros((N, P), dtype=float)

for i, row in enumerate(labelled_set):

    logger.info("Reading labelled set spectra ({}/{})".format(i + 1, N))
    
    filename = os.path.join(
        CANNON_DATA_DIR, 
        row["TELESCOPE"],
        row["LOCATION_ID"],
        "{}.pkl".format(row["APOGEE_ID"]))
    
    with open(filename, "rb") as fp:
        flux, ivar = pickle.load(fp)

    normalized_flux[i, :] = flux
    normalized_ivar[i, :] = ivar

# TODO: Cache the normalized_flux and normalized_ivar into a single file?


#-------------------#
# 3. Train a model. #
#-------------------#

model = tc.L1RegularizedCannonModel(labelled_set, normalized_flux,
    normalized_ivar, dispersion, threads=THREADS)

model.vectorizer = tc.vectorizer.NormalizedPolynomialVectorizer(labelled_set, 
    tc.vectorizer.polynomial.terminator(LABEL_NAMES, MODEL_ORDER))

model.regularization = MODEL_REGULARIZATION
model.s2 = 0

model.train()
model._set_s2_by_hogg_heuristic()

model_filename = os.path.join(CANNON_MODEL_DIR, "{}.pkl".format(MODEL_NAME))
model.save(model_filename, include_training_data=True, overwrite=clobber_model)

# TODO: Make some one-to-one plots to show sensible ness.
# TODO: Automatically run crossvalidation?


#---------------------------------------------------#
# 4. Generate a script to test the stacked spectra. #
#---------------------------------------------------#
stacked_spectra = [o for (s, i, o) in normalized_result if s in (None, True)]
stacked_spectra_path = os.path.join(CANNON_DATA_DIR, "stacked-spectra.list")
with open(stacked_spectra_path, "w") as fp:
    fp.write("\n".join(stacked_spectra))

logger.info(
    "The following command will perform the test step on all stacked spectra:"\
    'tc fit "{model_filename}" --from-filename "{spectrum_list}" -t {threads}'\
    .format(model_filename=model_filename, spectrum_list=stacked_spectra_path,
        threads=THREADS))

# TODO: initial positions?

#-----------------------------------------------------------------#
# 5. Make travel arrangements to accept Nobel Prize in Stockholm. # 
#-----------------------------------------------------------------#

