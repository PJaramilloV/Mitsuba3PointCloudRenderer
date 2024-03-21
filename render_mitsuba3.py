from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData
from tqdm import tqdm
import numpy as np
import argparse
import mitsuba
import glob
import os

from utils import (
    standardize_bbox, colormap, 
    get_files, rreplace, get_images,
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
            <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
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
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""


def main(pathToFile, num_points_per_object, forced=False):
    filename, file_extension = os.path.splitext(pathToFile)
    folder = os.path.dirname(pathToFile)
    filename = os.path.basename(pathToFile)

    # for the moment supports npy and ply
    if (file_extension == '.npy'):
        pclTime = np.load(pathToFile)
        pclTimeSize = np.shape(pclTime)
    elif (file_extension == '.npz'):
        pclTime = np.load(pathToFile)
        pclTime = pclTime['pred']
        pclTimeSize = np.shape(pclTime)
    elif (file_extension == '.ply'):
        ply = PlyData.read(pathToFile)
        vertex = ply['vertex']
        (x, y, z) = (vertex[t] for t in ('x', 'y', 'z'))
        pclTime = np.column_stack((x, y, z))
    else:
        print('unsupported file format.')
        return

    if (len(np.shape(pclTime)) < 3):
        pclTimeSize = [1, np.shape(pclTime)[0], np.shape(pclTime)[1]]
        pclTime.resize(pclTimeSize)

    for pcli in range(0, pclTimeSize[0]):
        pcl = pclTime[pcli, :, :]
        
        pcl = standardize_bbox(pcl, num_points_per_object)
        pcl = pcl[:, [2, 0, 1]]
        pcl[:, 0] *= -1
        pcl[:, 2] += 0.0125

        xml_segments = [xml_head]
        for i in range(pcl.shape[0]):
            color = colormap(pcl[i, 0] + 0.5, pcl[i, 1] + 0.5, pcl[i, 2] + 0.5 - 0.0125)
            xml_segments.append(xml_ball_segment.format(pcl[i, 0], pcl[i, 1], pcl[i, 2], *color))
        xml_segments.append(xml_tail)

        xml_content = str.join('', xml_segments)
        object_name = rreplace(filename,'.','_')

        xmlFile = os.path.join(folder, f"{object_name}_{pcli:02d}_{num_points_per_object}.xml")
        debug_msg(['Writing to: ', xmlFile])

        with open(xmlFile, 'w') as f:
            f.write(xml_content)
        f.close()
        
        png_file = os.path.join(folder, f"{object_name}_{pcli:02d}_{num_points_per_object}.png")
        if forced or (not os.path.exists(png_file)):
            debug_msg(['Running Mitsuba, loading: ', xmlFile])
            scene = mitsuba.load_file(xmlFile)
            render = mitsuba.render(scene)
            debug_msg(['writing to: ', png_file])
            mitsuba.util.write_bitmap(png_file, render)
        else:
            debug_msg('skipping rendering because the file already exists')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="filename or pattern to look for npy/ply")
    parser.add_argument("-n", "--num_points_per_object", type=int, default=2048)
    parser.add_argument("-v", "--mitsuba_variant", type=str, choices=mitsuba.variants(), default="scalar_rgb")
    parser.add_argument("-j", "--join_renders", type=bool, default=True)
    parser.add_argument("-c", "--clear", type=bool, default=False, help='clear all previous images corresponding to the files')
    parser.add_argument('-f', '--force_render', type=bool, default=False)
    parser.add_argument('-d', '--debug', type=bool, default=False)
    return parser.parse_args()


def merge_renders(renders, filename):
    images = []
    font = ImageFont.truetype("FreeSans.ttf", 30)
    combined_size = {'width': 0, 'height': 0}
    for render in renders:
        image = Image.open(render)
        factor = 540 / image.height
        image = image.resize((int(factor * image.width), 540))
        file = render.split('/')[-1]

        # Draw text
        ImageDraw.Draw(image).text((30, 10), file, fill=(128, 128, 128), font=font)

        images.append(image)
        combined_size['width'] = max(combined_size['width'], images[-1].width)
        combined_size['height'] += images[-1].height

    combined_image = Image.new("RGB", (combined_size["width"], combined_size["height"]))

    h = 0
    for image in images:
        combined_image.paste(image, (0, h))
        h += image.height
        image.close()

    print("Saving", filename)
    combined_image.save(filename)

if __name__ == "__main__":
    args = parse_args()
    mitsuba.set_variant(args.mitsuba_variant)
    files = get_files(args.filename)
    if args.debug:
        debug_msg = print

    if args.clear:
        imgs = get_images(files, replace_dots=True)
        for img in imgs:
            xml = img.replace('png', 'xml')
            os.remove(xml)
            os.remove(img)

    for path in tqdm(files):
        main(path, args.num_points_per_object, forced=args.force_render)

    if args.join_renders:
        deepest_dir:str = args.filename
        try:
            first_wild = deepest_dir.index('*')
            deepest_dir = deepest_dir[first_wild]
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
        merge_renders(imgs, os.path.join(deepest_dir,'data_view.png'))

