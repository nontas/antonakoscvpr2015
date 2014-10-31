from menpo.image import Image


def build_patches_image(image, centres, patch_shape):
    r"""
    Return the image patches as a menpo.image object
    size: patch_shape[0] x patch_shape[1] x n_channels x n_points
    """
    # extract patches
    if centres is None:
        patches = image.extract_patches_around_landmarks(patch_size=patch_shape,
                                                         as_single_array=True)
    else:
        patches = image.extract_patches(centres, patch_size=patch_shape,
                                        as_single_array=True)

    # build patches image
    return Image(patches)


def vectorize_patches_image(patches_image):
    r"""
    Return the image patches vector by calling the menpo.image.Image.as_vector()
    size: (patch_shape[0] * patch_shape[1] * n_channels * n_points) x 1
    """
    return patches_image.as_vector()
