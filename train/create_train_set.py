import os
import cv2
import json
import zipfile
import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from pathlib import Path

THICKNESS = 6
TARGET_SHAPE = (1680, 1024)
IN_DIR, OUT_DIR = 'D:/verwerkt', '../train_data'


def generate_line_mask(lines, mask_shape):
    mask = np.zeros(mask_shape, dtype='uint8')
    for line in lines:
        cv2.line(mask, (int(line[0][0]), int(line[0][1])), (int(line[1][0]), int(line[1][1])), 255, thickness=THICKNESS)

    mask = cv2.resize(mask, (TARGET_SHAPE[1], TARGET_SHAPE[0])).astype(np.uint8)
    mask = np.nonzero(mask)

    return mask


def generate_lines_from_json(obs, image_shape):
    lines, points_dict = obs['lines'], obs['points']

    lines_list = []
    for line, points in lines.items():
        for i, end in enumerate(points['points'][1:]):
            start = points['points'][i]
            lines_list.append([points_dict[start]['position'], points_dict[end]['position']])

    masks = [generate_line_mask(lines_list, image_shape)]

    return masks


def generate_building_from_json(obs, image_shape):
    points_dict, buildings = obs['points'], obs['buildings']

    g = nx.Graph()
    for line in obs['lines'].values():
        line_points = line['points']
        for i in range(1, len(line_points)):
            segment = (line_points[i - 1], line_points[i])
            g.add_nodes_from(segment)
            g.add_edge(*segment)

    for point in obs['points'].keys():
        if not g.has_node(point):
            g.add_node(point)

    buildings_list = []
    for building, building_points in buildings.items():
        building_points = building_points['points']
        for i in range(1, len(building_points)):
            try:
                path = nx.shortest_path(g, building_points[i - 1], building_points[i])
                for k in range(1, len(path)):
                    buildings_list.append([path[k - 1], path[k]])
            except nx.exception.NetworkXNoPath:
                buildings_list.append([building_points[i - 1], building_points[i]])

    buildings_list = list(map(lambda x: [points_dict[x[0]]['position'], points_dict[x[1]]['position']], buildings_list))
    masks = [generate_line_mask(buildings_list, image_shape)]

    return masks


def get_masked(img, line_masks, building_masks, visualize=False):
    mask_img = np.zeros(img.shape, dtype=np.uint8)

    if line_masks is not None:
        for mask in line_masks:
            mask_img[mask] = 255

    if building_masks is not None:
        for mask in building_masks:
            mask_img[mask] = 255

    if visualize:
        plt.figure(figsize=(4, 4))
        plt.imshow(mask_img)
        plt.show()

    return mask_img


def read_zip(zip_name, line_masks=True, building_masks=False):
    with zipfile.ZipFile(zip_name, 'r') as archive:
        prefix, postfix = 'observations/snapshots/latest/', '.latest.json'
        sketch_files = list(filter(lambda x: x.startswith(prefix) and x.endswith(postfix), archive.namelist()))
        for i, sketch_file in enumerate(sketch_files):
            sketch_name = sketch_file[len(prefix):-len(postfix)]

            images_dir = os.path.join(OUT_DIR, 'images')
            Path(images_dir).mkdir(parents=True, exist_ok=True)

            masks_dir = os.path.join(OUT_DIR, 'masks')
            Path(masks_dir).mkdir(parents=True, exist_ok=True)

            attachment_prefix, img_extension = 'observations/attachments/front/' + sketch_name, '.JPG'
            image_files = list(filter(lambda x: x.startswith(attachment_prefix) and x.endswith(img_extension),
                                      archive.namelist()))

            if len(image_files):
                logging.info('Processing sketch: {index}: {name}.'.format(index=i, name=sketch_name))

                file = archive.read(image_files[0])
                image = cv2.imdecode(np.frombuffer(file, np.uint8), 1)
                image_shape = image.shape[:2]
                image = cv2.resize(image, (TARGET_SHAPE[1], TARGET_SHAPE[0])).astype(np.uint8)

                with archive.open(sketch_file, 'r') as fh:
                    json_data = json.loads(fh.read())

                image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                if line_masks:
                    l_masks = generate_lines_from_json(json_data, image_shape)
                    if not l_masks[0][0].size:
                        logging.warning('Skipping sketch: no lines found.')
                        continue
                else:
                    l_masks = None

                if building_masks:
                    b_masks = generate_building_from_json(json_data, image_shape)
                else:
                    b_masks = None

                masked_img = get_masked(image_gs, l_masks, b_masks, visualize=False)

                cv2.imwrite(os.path.join(images_dir, sketch_name + '.tif'), image_gs)
                cv2.imwrite(os.path.join(masks_dir, sketch_name + '.tif'), masked_img)
            else:
                logging.info('SKIPPING: {index}: {name}.'.format(index=i, name=sketch_name))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s;%(levelname)s;%(message)s')

    for j, name in enumerate(os.listdir(IN_DIR)):
        logging.info('Processing project: {index}: {name}.'.format(index=j, name=name))
        read_zip(os.path.join(IN_DIR, name), line_masks=False, building_masks=True)
