__all__ = [
    'add_noise_in_pixels',
    'smear_light_in_pixels',
]

import numpy as np

def add_noise_in_pixels(rng, image, extra_noise_in_dim_pixels,
                        extra_bias_in_dim_pixels, transition_charge,
                        extra_noise_in_bright_pixels):
    """

    Parameters
    ----------
    rng: numpy.random.default_rng  random number generator

    image: charges (p.e.) in the camera

    To be tuned by comparing the starting MC and data:

    extra_noise_in_dim_pixels: mean additional number of p.e. to be added (
    Poisson noise) to pixels with charge below transition_charge

    extra_bias_in_dim_pixels: mean bias (w.r.t. original charge) of the new
    charge in pixels. Should be 0 for non-peak-search pulse integrators

    transition_charge: border between "dim" and "bright" pixels

    extra_noise_in_bright_pixels: mean additional number of p.e. to be added (
    Poisson noise) to pixels with charge above transition_charge. This is
    unbiased, i.e. Poisson noise is introduced, and its average subtracted,
    so that the mean charge in bright pixels remains unaltered. This is
    because we assume that above transition_charge the integration window
    is determined by the Cherenkov light, and would not be modified by the
    additional NSB noise (presumably small compared to the C-light)

    Returns
    -------
    Modified (noisier) image

    """

    qcopy = image.copy()
    image[qcopy < transition_charge] += (rng.poisson(
            extra_noise_in_dim_pixels, (qcopy < transition_charge).sum()) -
                                         extra_noise_in_dim_pixels +
                                         extra_bias_in_dim_pixels)
    image[qcopy > transition_charge] += (rng.poisson(
            extra_noise_in_bright_pixels, (qcopy > transition_charge).sum())
                                         - extra_noise_in_bright_pixels)
    return image


def smear_light_in_pixels(image, camera_geom, smeared_light_fraction):
    """

    Parameters
    ----------
    image: charges (p.e.) in the camera

    camera_geom: camera geometry

    smeared_light_fraction: fraction of the light in a pixel that will be
    distributed equally among its immediate surroundings, i.e. immediate
    neighboring pixels. Some light is lost for pixels which are at the
    camera edge and hence don't have all possible neighbors

    Returns
    -------
    Modified (smeared) image

    """

    # number of neighbors of completely surrounded pixels:
    max_neighbors = np.max([len(pixnbs) for pixnbs in camera_geom.neighbors])

    # Move a fraction of the light in each pixel (fraction) into its neighbors,
    # to simulate a worse PSF:
    q_smeared = (image * camera_geom.neighbor_matrix *
                 smeared_light_fraction / max_neighbors)
    # Light remaining in pixel:
    q_remaining = image * (1 - smeared_light_fraction)
    image = q_remaining + np.sum(q_smeared, axis=1)
