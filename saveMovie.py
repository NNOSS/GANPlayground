

import imageio

def writeMovie(path,images):
    with imageio.get_writer(path, mode='I') as writer:
        for image in images:
            writer.append_data(image)
