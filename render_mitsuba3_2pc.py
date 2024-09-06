import os.path
import glob
from PIL import Image, ImageDraw

from render_mitsuba3 import *


def read_ply(path):
    ply = PlyData.read(path)
    vertex = ply['vertex']
    (x, y, z) = (vertex[t] for t in ('x', 'y', 'z'))
    (r, g, b) = (vertex[t] for t in ('red', 'green', 'blue'))

    pcl_time = np.column_stack((x, y, z))
    pcl_time_rgb = np.column_stack((r, g, b))
    if len(np.shape(pcl_time)) < 3:
        new_size = (1,) + np.shape(pcl_time)
        pcl_time.resize(new_size)
        pcl_time_rgb.resize(new_size)

    return pcl_time, pcl_time_rgb


def read_obj(path):
    import pywavefront
    
    evaluated_vertices = pywavefront.Wavefront(path)
    evaluated_vertices = np.array(evaluated_vertices.vertices)
    
    pcl_time_rgb = evaluated_vertices[:, 3:]
    pcl_time = evaluated_vertices[:, :3]
    
    if len(np.shape(pcl_time)) < 3:
        new_size = (1,) + np.shape(pcl_time)
        pcl_time.resize(new_size)
        if pcl_time_rgb.shape[1] == 3:
            pcl_time_rgb.resize(new_size)
        else:
            # no color info
            pcl_time_rgb = None
    
    return pcl_time, pcl_time_rgb


def standardize_bbox2(pcl1, pcl2, points_per_object):
    """
    Standardizes to [-0.5, 0.5], but considering the union of pcl1 and pcl2, so the standardization is not independent

    :points_per_object: Total points, it will be weighted.
    """
    print(f"standardize to {points_per_object} points")

    # weight the points per object (ppo)
    total_n = pcl1.shape[0] + pcl2.shape[0]
    pcl1_ppo = int(pcl1.shape[0] * (points_per_object / total_n))
    pcl2_ppo = points_per_object - pcl1_ppo

    # sample pcl1
    pt1_indices = np.random.choice(pcl1.shape[0], pcl1_ppo, replace=False)
    np.random.shuffle(pt1_indices)
    pcl1 = pcl1[pt1_indices]
    # sample pcl2
    pt2_indices = np.random.choice(pcl2.shape[0], pcl2_ppo, replace=False)
    np.random.shuffle(pt2_indices)
    pcl2 = pcl2[pt2_indices]

    pcl_concat = np.concatenate((pcl1, pcl2), axis=0)
    mins = np.amin(pcl_concat, axis=0)
    maxs = np.amax(pcl_concat, axis=0)
    center = (mins + maxs) / 2.
    scale = np.amax(maxs - mins)
    print("Center: {}, Scale: {}".format(center, scale))

    # [-0.5, 0.5]
    pcl1 = ((pcl1 - center) / scale).astype(np.float32)  # [-0.5, 0.5]
    pcl2 = ((pcl2 - center) / scale).astype(np.float32)  # [-0.5, 0.5]
    return pcl1, pcl2


def main2(partial_file_path, evaluated_file_path, out_file_path, num_points_per_object):
    filename, out_file_extension = os.path.splitext(out_file_path)
    _, in_file_extension = os.path.splitext(partial_file_path)
    # filename = os.path.basename(filename)
    # folder = os.path.dirname(partial_file_path)
    
    if in_file_extension == '.ply':
        partial_pcl_time, _ = read_ply(partial_file_path)
        evaluated_pcl_time, color_evaluated_pcl_time = read_ply(evaluated_file_path)
    elif in_file_extension == '.obj':
        partial_pcl_time, _ = read_obj(partial_file_path)
        evaluated_pcl_time, color_evaluated_pcl_time = read_obj(evaluated_file_path)
    else:
        print(f"unsupported {in_file_extension} file")
        return

    for pcli in range(0, np.shape(partial_pcl_time)[0]):
        partial_pcl = partial_pcl_time[pcli, :, :]
        evaluated_pcl = evaluated_pcl_time[pcli, :, :]
        if color_evaluated_pcl_time is not None:
            # filter black and dark-gray points in evaluated_pcl
            indices = np.where(color_evaluated_pcl_time[pcli][:, 0] >= 10)[0]
            evaluated_pcl = evaluated_pcl[indices]

        # standardize both pcl
        partial_pcl, evaluated_pcl = standardize_bbox2(partial_pcl, evaluated_pcl, num_points_per_object)

        def add_xml_segments(pcl, colormap_fun=colormap):
            pcl = pcl[:, [2, 0, 1]]
            pcl[:, 0] *= -1
            pcl[:, 2] += 0.0125

            for i in range(pcl.shape[0]):
                color = colormap_fun(pcl[i, 0] + 0.5, pcl[i, 1] + 0.5, pcl[i, 2] + 0.5 - 0.0125)
                xml_segments.append(xml_ball_segment.format(pcl[i, 0], pcl[i, 1], pcl[i, 2], *color))

        def colormap1(x, y, z):
            vec = np.array([z, 0, .05 * x])
            vec = np.clip(vec, 0.001, 1.0)
            norm = np.sqrt(np.sum(vec ** 2))
            vec /= norm
            return vec

        def colormap2(x, y, z):
            vec = np.array(
                [-.3 * z, z, -.3 * z + .3 * x])  # dark green, using a little of red in x for a little color effect
            vec = np.clip(vec, 0.001, 1.0)
            norm = np.sqrt(np.sum(vec ** 2))
            vec /= norm
            return vec

        xml_segments = [xml_head]
        add_xml_segments(partial_pcl, colormap1)
        add_xml_segments(evaluated_pcl, colormap2)
        xml_segments.append(xml_tail)

        xml_content = str.join('', xml_segments)

        xmlFile = f"{filename}_{pcli:02d}.xml"  # os.path.join(folder, f"{filename}_restored_{pcli:02d}.xml")
        print(['Writing to: ', xmlFile])

        with open(xmlFile, 'w') as f:
            f.write(xml_content)
        f.close()

        png_file = f"{filename}_{pcli:02d}" + out_file_extension  # os.path.join(folder, f"{filename}_restored_{pcli:02d}.png")
        print(['Running Mitsuba, loading: ', xmlFile])
        scene = mitsuba.load_file(xmlFile)
        render = mitsuba.render(scene)
        print(['writing to: ', png_file])
        mitsuba.util.write_bitmap(png_file, render, write_async=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("basename",
                        help="basename of the .ply partial and evaluated files, or name of the folder to process")
    parser.add_argument("-n", "--num_points_per_object", type=int, default=2048)
    parser.add_argument("-v", "--mitsuba_variant", type=str, choices=mitsuba.variants(), default="scalar_rgb")
    parser.add_argument('-u', '--up_axis', type=str, choices=['x','y','z'], default='z', help='Axis considered height')

    parser.add_argument("--partial_suffix", type=str, default="_points.partial.ply")
    parser.add_argument("--evaluated_suffix", type=str, default="_evaluated_pc.ply")
    parser.add_argument("--restored_suffix", type=str, default="_restored_pc.png")

    return parser.parse_args()


def get_pairs_in_folder(partial_suffix, evaluated_suffix, pattern):
    pairs = []

    partial_files = glob.glob(pattern + partial_suffix)
    for partial_ply_path in partial_files:
        evaluated_ply_path = partial_ply_path.rsplit(partial_suffix, 1)[0] + evaluated_suffix
        if os.path.exists(evaluated_ply_path):
            pairs.append([partial_ply_path, evaluated_ply_path])

    return pairs


def get_restored_renders(pattern):
    pattern, ext = os.path.splitext(pattern)
    pattern = pattern + "*" + ext  # To look for stuff like restored_pc_01.png, not just restored_pc.png

    restored_files = sorted(glob.glob(pattern))
    return restored_files


if __name__ == "__main__":
    args = parse_args()
    mitsuba.set_variant(args.mitsuba_variant)
    
    up = args.up_axis
    up_x, up_y, up_z = ('x' in up), ('y' in up), ('z' in up)
    up_vec = f'{1&up_x},{1&up_y},{1&up_z}'
    xml_head = xml_head.format(up_vec)
    xml_tail = xml_tail.format(up_vec)
    
    pair_paths = get_pairs_in_folder(args.partial_suffix, args.evaluated_suffix, args.basename)
    for p in pair_paths:
        print(p[0], p[1], p[0].rsplit(args.partial_suffix, 1)[0] + args.restored_suffix)
        try:
            main2(p[0],
                  p[1],
                p[0].rsplit(args.partial_suffix, 1)[0] + args.restored_suffix,
                args.num_points_per_object)
        except Exception as e:
            print(e)

    if len(pair_paths) > 1:
        merge_renders(get_restored_renders(args.basename+args.restored_suffix), args.basename[:-1] + "combined.png")