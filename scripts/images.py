from PIL import Image
import polars as pl
from uuid import uuid4
import os
import numpy as np


def saving_sample_wrapper(
    image_generator,
    output_folder: str,
    initial_sample_id: int,
    initial_timestamp: float,
):
    """this wrapper saves generated images and creates target and
    mapping csv files."""
    sample_id = initial_sample_id
    timestamp = initial_timestamp

    # while True:
    elem = next(image_generator)
    image_name_list = []
    label_list = []
    sample_id_list = []
    timestamp_list = []

    for i, image in enumerate(elem[0]):
        name = str(uuid4())
        pil_image = Image.fromarray(np.uint8(image * 255).squeeze())
        pil_image.save(os.path.join(output_folder, 'images', name + '.png'))
        if int(elem[1][i]) == 0:
            pil_image.save(
                os.path.join(
                    output_folder, 'tf_dirs', 'ok_front', name + '.png'
                )
            )
        else:
            pil_image.save(
                os.path.join(
                    output_folder, 'tf_dirs', 'def_front', name + '.png'
                )
            )
        image_name_list.append(name)
        sample_id_list.append(sample_id)
        timestamp_list.append(timestamp)

        # add 10 seconds
        timestamp += 10
        sample_id += 1

    for label in elem[1]:
        label_list.append(int(label))

    target = pl.DataFrame(
        {
            'label': label_list,
            'sample_id': sample_id_list,
            'timestamp': timestamp_list,
        }
    )
    mapping = pl.DataFrame(
        {
            'file_name': list(map(lambda x: x + '.png', image_name_list)),
            'sample_id': sample_id_list,
            'timestamp': timestamp_list,
        }
    )
    with open(os.path.join(output_folder, 'target.csv'), mode='ab') as f:
        target.write_csv(f, include_header=False)
    with open(os.path.join(output_folder, 'mapping.csv'), mode='ab') as f:
        mapping.write_csv(f, include_header=False)

    yield elem
