
"""
APOGEE DR14 processing script for The Cannon.
"""

__author__ = [
    "Andy Casey <andy.casey@gmail.com>",
    "Gail Zasowski <gail.zasowski@gmail.com>"
]

from glob import glob

import .continuum



#------------------------------------------------------#
# 0. Download all the data and set required variables. #
#------------------------------------------------------#

PICKLE_PROTOCOL = -1
ORIGINAL_DATA_DIR = "/"
CANNON_DATA_DIR = "/"



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
for i, input_path in enumerate(individual_visit_paths):

    logger.info("({}/{}): {}".format(i, N, input_path))

    # Generate the output path for The Cannon-normalized spctra.
    apogee_id = input_path.split("-r7-")[-1][:-4]
    telescope, location_id = input_path.split("/")[-3:-1]

    output_path = os.path.join(
        CANNON_DATA_DIR, telescope, location_id, "{}.pkl".format(apogee_id))

    if os.path.exists(output_path) and not clobber_normalization:
        logger.info("Skipping normalization of {}..".format(input_path))
        continue

    try:
        stacked, visits, metadata = normalize_individual_visits(
            input_path, full_output=True, **normalization_kwds)

    except:
        logger.exception("Normalization failed on {}".format(input_path))
        continue

    metadata.update(APOGEE_ID=apogee_id)

    stacked = np.vstack(stacked)
    visits = np.vstack(visits)
    
    with open(output_path, "wb") as fp:
        pickle.dump(stacked, fp, PICKLE_PROTOCOL)

    with open("{}.visits".format(output_path), "wb") as fp:
        pickle.dump(visits, fp, PICKLE_PROTOCOL)

    with open("{}.meta".format(output_path), "wb") as fp:
        pickle.dump(metadata, fp, PICKLE_PROTOCOL)



# 3. Construct a training set from the stacked, individual visit spectra.

# 4. Train a model.

# 5. Test a model on the individual visit spectra (stacked and unstacked).

# 6. Collect test results into a single file.

# 7. Make arrangements to accept Nobel Prize in Stockholm.
