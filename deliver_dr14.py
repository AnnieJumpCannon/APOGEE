
"""
APOGEE DR14 processing script for The Cannon.
"""

__author__ = [
    "Andy Casey <andy.casey@gmail.com>",
    "Gail Zasowski <gail.zasowski@gmail.com>"
]

import cPickle as pickle
import logging
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
from astropy.io import fits
from astropy.table import Table
from glob import glob

import AnniesLasso as tc

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
LABELLED_SET_PATH = os.path.join(
    CANNON_MODEL_DIR, "apogee-dr14-giants-training-set.fits")
MODEL_NAME = "apogee-dr14-giants"

# The label names to use in the model.
MODEL_LABEL_NAMES = ("TEFF", "LOGG", "FE_H", "MG_H", "AL_H")
MODEL_ORDER = 2
MODEL_SCALE_FACTOR = 1.0
MODEL_REGULARIZATION = 0.0

# Danger Will Robinson!
SKIP_NORMALIZATION = True
SKIP_TRAINING = True


#------------------------------------------------------#
# 1. Normalize and stack the individual visit spectra. #
#------------------------------------------------------#
if not SKIP_NORMALIZATION:

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

    individual_visit_paths = \
        glob("{}/apo*/*/apStar*.fits".format(ORIGINAL_DATA_DIR))
    N_individual_visits = len(individual_visit_paths)

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
        "Starting {} threads to do pseudo-continuum-normalization of {} stars"\
        .format(THREADS, N_individual_visits))

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
                dispersion = 10**(image[1].header["CRVAL1"] \
                    + np.arange(image[1].data.size) * image[1].header["CDELT1"])
            
            with open(os.path.join(CANNON_DATA_DIR, "dispersion.pkl"), "wb") as fp:
                pickle.dump(dispersion, fp, PICKLE_PROTOCOL)
            
            # We only need to save the dispersion once: break the loop.
            break

else:
    logger.warn("Skipping normalization!")
    normalized_result = [(None, "", output_path) \
        for output_path in glob(os.path.join(CANNON_DATA_DIR, "*", "*", "*.pkl"))]

#-------------------------------------------------------#
# 2. Construct a training set from the stacked spectra. #
#-------------------------------------------------------#
model_filename = os.path.join(CANNON_MODEL_DIR, "{}.model".format(MODEL_NAME))

if not SKIP_TRAINING:

    clobber_model = True
    labelled_set = Table.read(LABELLED_SET_PATH)
    N_labelled = len(labelled_set)

    # TODO: something's wrong with our dispersion that we extracted.
    #with open(os.path.join(CANNON_DATA_DIR, "dispersion.pkl"), "rb") as fp:
    #    dispersion = pickle.load(fp)
    #P = dispersion.size
    dispersion = None
    P = 8575 # MAGIC

    # These defaults (flux = 1, ivar = 0) will mean that even if we don't find a
    # spectrum for a single star in the training set, then that star will just have
    # no influence on the training (since ivar = 0 implies infinite error on flux).

    normalized_flux = np.ones((N_labelled, P), dtype=float)
    normalized_ivar = np.zeros((N_labelled, P), dtype=float)

    for i, row in enumerate(labelled_set):

        logger.info(
            "Reading labelled set spectra ({}/{})".format(i + 1, N_labelled))

        filename = os.path.join(
            CANNON_DATA_DIR, 
            row["TELESCOPE"],
            str(row["LOCATION_ID"]),
            "{}.pkl".format(row["APOGEE_ID"]))
        
        if not os.path.exists(filename):
            logger.warn("Could not find filename for labelled set star {}: {}"\
                .format(row["APOGEE_ID"], filename))
            continue

        with open(filename, "rb") as fp:
            flux, ivar = pickle.load(fp)

        normalized_flux[i, :] = flux
        normalized_ivar[i, :] = ivar

    # TODO: Cache the normalized_flux and normalized_ivar into a single file so that
    #       it is faster to read in next time?
    assert  np.isfinite(normalized_flux).all(), \
            "Non-finite values in normalized_flux!"
    assert  np.isfinite(normalized_ivar).all(), \
            "Non-finite values in normalized_ivar!"

    # Exclude labelled set stars where there is no spectrum, only because it
    # will get annoying later on when we are doing 1-to-1 and cross-validation
    keep = np.any(normalized_ivar > 0, axis=1)
    if not np.all(keep):
        logger.info(
            "Excluding {} labelled set stars where there was no information in "
            "the spectrum".format(np.sum(~keep)))
        labelled_set = labelled_set[keep]
        normalized_flux = normalized_flux[keep]
        normalized_ivar = normalized_ivar[keep]

    #---------------------------------#
    # 3. Construct and train a model. #
    #---------------------------------#
    model = tc.L1RegularizedCannonModel(
        labelled_set, normalized_flux, normalized_ivar, dispersion, 
        threads=THREADS)

    model.vectorizer = tc.vectorizer.NormalizedPolynomialVectorizer(
        labelled_set, 
        tc.vectorizer.polynomial.terminator(MODEL_LABEL_NAMES, MODEL_ORDER),
        scale_factor=MODEL_SCALE_FACTOR)

    model.s2 = 0
    model.regularization = MODEL_REGULARIZATION

    model.train()
    model._set_s2_by_hogg_heuristic()

    model.save(
        model_filename, include_training_data=False, overwrite=clobber_model)

else:
    assert  os.path.exists(model_filename), \
            "Are you sure you meant to skip the training?"
    model = tc.load_model(model_filename, threads=THREADS)


# Make some 1-to-1 plots just to show sensible behaviour.
X = model.labels_array()
Y = model.fit(model.normalized_flux, model.normalized_ivar)

for i, label_name in enumerate(MODEL_LABEL_NAMES):

    x = X[:, i]
    y = Y[:, i]

    fig, ax = plt.subplots()
    ax.scatter(x, y, facecolor="#000000", alpha=0.5)

    lims = np.array([ax.get_xlim(), ax.get_ylim()])
    lims = (lims.min(), lims.max())
    ax.plot(lims, lims, c="#666666", zorder=-1, linestyle=":")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("Labelled")
    ax.set_ylabel("Inferred")

    mean, rms = np.nanmean(y - x), np.nanstd(y - x)
    title = "{}: ({:.2f}, {:.2f})".format(label_name, mean, rms)
    ax.set_title(title)
    logger.info("Mean and RMS for {}".format(title))

    figure_path = os.path.join(CANNON_MODEL_DIR, "{}-{}-1to1.png".format(
        MODEL_NAME, label_name))
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)

    logger.info(
        "Created 1-to-1 figure for {} at {}".format(label_name, figure_path))


#---------------------------------------------------#
# 4. Generate a script to test the stacked spectra. #
#---------------------------------------------------#
stacked_spectra = [o for (s, i, o) in normalized_result if s in (None, True)]
stacked_spectra_path = os.path.join(CANNON_DATA_DIR, "stacked-spectra.list")
with open(stacked_spectra_path, "w") as fp:
    fp.write("\n".join(stacked_spectra))

# Create a file of initial positions that we will use at test time.
# Here we will just use the mean label value.
initial_path = os.path.join(CANNON_DATA_DIR, "initial_labels.txt")
np.savetxt(initial_path, np.mean(model.labels_array, axis=0).reshape(-1, 1))
logger.info(
    "Initial positions to try at test time saved to {}".format(initial_path))

logger.info(
    """The following commands will perform the test step on all stacked spectra:
    cd "{data_dir}"
    tc fit "{model_filename}" --from-filename "{spectrum_list}" -t {threads}"""\
    .format(model_filename=model_filename, spectrum_list=stacked_spectra_path,
        threads=THREADS, data_dir=CANNON_DATA_DIR))

expected_results_path = os.path.join(CANNON_DATA_DIR, "stacked-results.list")
expected_results = \
    [o.replace(".pkl", "-result.pkl") for (s, i, o) in normalized_result \
        if s in (None, True)]
with open(expected_results_path, "w") as fp:
    fp.write("\n".join(expected_results))

logger.info(
    """After the test step, the following command will collect all results into
    a single table:

    cd "{data_dir}"
    tc join {model_name}-catalog.fits --from-filename "{expected_results_path}" --errors --clobber"""\
    .format(data_dir=CANNON_DATA_DIR, model_name=MODEL_NAME, 
        expected_results_path=expected_results_path))

logger.info("Fin.")

#-----------------------------------------------------------------#
# 5. Make travel arrangements to accept Nobel Prize in Stockholm. # 
#-----------------------------------------------------------------#