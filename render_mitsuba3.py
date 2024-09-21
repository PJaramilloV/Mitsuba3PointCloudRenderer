import argparse
import os

import mitsuba
import numpy as np
from tqdm import tqdm

from plyfile import PlyData
from utils import (
    standardize_bbox, colormap,
    get_files, rreplace, get_images, merge_renders,
    write_xml, render_xml,
    debug_msg
)

# replaced by command line arguments
# PATH_TO_NPY = 'pcl_ex.npy' # the tensor to load

# note that sampler is changed to 'independent' and the ldrfilm is changed to hdrfilm
xml_head = \
    """
<scene version="3.0.0">
    <integrator type="path">
        <integer name="max_depth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="far_clip" value="100"/>
        <float name="near_clip" value="0.1"/>
        <transform name="to_world">
            <lookat origin="{}" target="0,0,0" up="{}"/>
        </transform>
        <float name="fov" value="25"/>
        <sampler type="independent">
            <integer name="sample_count" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="1920"/>
            <integer name="height" value="1080"/>
            <rfilter type="gaussian"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surface_material">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="int_ior" value="1.46"/>
        <rgb name="diffuse_reflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
    
"""

# I also use a smaller point size
xml_ball_segment = \
    """
    <shape type="sphere">
        <float name="radius" value="0.007"/>
        <transform name="to_world">
            <translate value="{}, {}, {}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""

xml_obj_segment = \
    """
    <shape type="obj">
        <string name="filename" value="{}"/>
    </shape>
"""

xml_tail = \
    """
    <shape type="rectangle">
        <ref name="bsdf" id="surface_material"/>
        <transform name="to_world">
            <scale value="10, 10, 1"/>
            <translate value="0, 0, -0.5"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="to_world">
            <scale value="10, 10, 1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="{}"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""


def main(pathToFile, num_points_per_object, forced=False, obj_as_pcl=False):
    filename, file_extension = os.path.splitext(pathToFile)
    folder = os.path.dirname(pathToFile)
    filename = os.path.basename(pathToFile)
    object_name = rreplace(filename, '.', '_')

    # for the moment supports npy and ply
    if file_extension == '.npy':
        pclTime = np.load(pathToFile)
        pclTimeSize = np.shape(pclTime)
    elif file_extension == '.npz':
        pclTime = np.load(pathToFile)
        pclTime = pclTime['pred']
        pclTimeSize = np.shape(pclTime)
    elif file_extension == '.ply':
        ply = PlyData.read(pathToFile)
        vertex = ply['vertex']
        (x, y, z) = (vertex[t] for t in ('x', 'y', 'z'))
        pclTime = np.column_stack((x, y, z))
    elif file_extension == '.obj':
        if obj_as_pcl:
            import pywavefront
            scene = pywavefront.Wavefront(pathToFile)
            if len(scene.materials) == 0:
                pclTime = np.array(scene.vertices)
            else:
                pclTime = []
                for _, material in scene.materials.items():
                    pclTime.append(np.array(material.vertices).reshape(-1, 3))
                pclTime = np.array(pclTime)
            pclTimeSize = np.shape(pclTime)
        else:
            xml_file = os.path.join(folder, f"{object_name}.xml")
            png_file = xml_file.replace('.xml', '.png')
            if forced or (not os.path.exists(png_file)):
                xml_segments = [xml_head, xml_obj_segment.format(pathToFile), xml_tail]
                xml_content = str.join('', xml_segments)
                write_xml(xml_file, xml_content)
                render_xml(xml_file, png_file)
            else:
                debug_msg('skipping rendering because the file already exists')
            return
    else:
        print('unsupported file format.')
        return

    if (len(np.shape(pclTime)) < 3):
        pclTimeSize = [1, np.shape(pclTime)[0], np.shape(pclTime)[1]]
        pclTime.resize(pclTimeSize)

    for pcli in range(0, pclTimeSize[0]):
        pcl = pclTime[pcli, :, :]

        pcl = standardize_bbox(pcl, min(num_points_per_object, pcl.shape[0]))
        pcl = pcl[:, [2, 0, 1]]
        pcl[:, 0] *= -1
        pcl[:, 2] += 0.0125

        xml_segments = [xml_head]
        for i in range(pcl.shape[0]):
            color = colormap(pcl[i, 0] + 0.5, pcl[i, 1] + 0.5, pcl[i, 2] + 0.5 - 0.0125)
            xml_segments.append(xml_ball_segment.format(pcl[i, 0], pcl[i, 1], pcl[i, 2], *color))
        xml_segments.append(xml_tail)

        xml_content = str.join('', xml_segments)

        xml_file = os.path.join(folder, f"{object_name}_{pcli:02d}_{num_points_per_object}.xml")
        png_file = xml_file.replace('.xml', '.png')

        if forced or (not os.path.exists(png_file)):
            write_xml(xml_file, xml_content)
            render_xml(xml_file, png_file)
        else:
            debug_msg('skipping rendering because the file already exists')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="filename or pattern to look for npy/ply")
    parser.add_argument("-n", "--num_points_per_object", type=int, default=2048)
    parser.add_argument("-v", "--mitsuba_variant", type=str, choices=mitsuba.variants(), default="scalar_rgb")
    parser.add_argument("-j", "--join_renders", type=eval, default=True)
    parser.add_argument('-k', '--keep_renders', type=eval, default=True,
                        help='keep rendered images after completing rendering')
    parser.add_argument('-u', '--up_axis', type=str, choices=['x', 'y', 'z'], default='z',
                        help='Axis considered height')
    parser.add_argument('-f', '--force_render', action='store_true', help='overwrite existing renders')
    parser.add_argument("-c", "--clear", action='store_true',
                        help='clear all previous images corresponding to the files')
    parser.add_argument('--render_obj_as_pointcloud', action='store_true')
    parser.add_argument('-d', '--debug', action='store_true')
    return parser.parse_args()


def remove_images(files):
    imgs = get_images(files, replace_dots=True)
    debug_msg(f'Removing {len(imgs)} images and xml files')
    for img in imgs:
        xml = img.replace('png', 'xml')
        os.remove(xml)
        os.remove(img)



if __name__ == "__main__":
    args = parse_args()
    mitsuba.set_variant(args.mitsuba_variant)
    files = get_files(args.filename)
    up = args.up_axis
    up_x, up_y, up_z = ('x' in up), ('y' in up), ('z' in up)
    up_vec = f'{1 & up_x},{1 & up_y},{1 & up_z}'
    origin_vec = '3,3,3'
    xml_head = xml_head.format(origin_vec, up_vec)
    xml_tail = xml_tail.format(up_vec)
    if args.debug:
        debug_msg = print

    if args.clear:
        remove_images(files)

    for path in tqdm(files):
        main(path, args.num_points_per_object, forced=args.force_render, obj_as_pcl=args.render_obj_as_pointcloud)

    if args.join_renders:
        deepest_dir: str = args.filename
        try:
            first_wild = deepest_dir.index('*')
            deepest_dir = deepest_dir[:first_wild]
            while not os.path.isdir(deepest_dir) and deepest_dir:
                path_parts = deepest_dir.split('/')[:-1]
                deepest_dir = "/".join(path_parts)
        except:
            # no wild_cards in directory
            path_parts = deepest_dir.split('/')[:-1]
            deepest_dir = "/".join(path_parts)

        if not deepest_dir:
            deepest_dir = '.'

        imgs = get_images(files, replace_dots=True)
        merge_renders(sorted(imgs), os.path.join(deepest_dir, 'data_view.png'))

    if not args.keep_renders:
        remove_images(files)
