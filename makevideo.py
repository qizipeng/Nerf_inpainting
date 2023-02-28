import numpy as np
import imageio
import os
from PIL import Image
from run_nerf_helpers import to8b


if __name__ == "__main__":
    rgb_path = "./logs/blender_paper_lego/renderonly_test_349999"
    rgbs =[]
    albedos=[]
    visits = []
    normals = []
    for i in range(25):
        img = Image.open(os.path.join(rgb_path, '{:03d}_lighting.png'.format(i)))
        rgbs.append(np.array(img))

        img = Image.open(os.path.join(rgb_path, '{:03d}_normal_pred.png'.format(i).format(i)))
        normals.append(np.array(img))

        img = Image.open(os.path.join(rgb_path, '{:03d}_albedo.png'.format(i)))
        albedos.append(np.array(img))

        img = Image.open(os.path.join(rgb_path, '{:03d}_visit.png'.format(i)))
        visits.append(np.array(img))


    rgbs = np.stack(rgbs, 0)
    imageio.mimwrite(os.path.join("./videos", 'RGBS_OL.mp4'), rgbs, fps=5, quality=8)

    normals = np.stack(normals, 0)
    imageio.mimwrite(os.path.join("./videos", 'NORMALS.mp4'), normals, fps=5, quality=8)

    albedos = np.stack(albedos, 0)
    imageio.mimwrite(os.path.join("./videos", 'ALBEDOS.mp4'), albedos, fps=5, quality=8)

    visits = np.stack(visits, 0)
    imageio.mimwrite(os.path.join("./videos", 'VISIT.mp4'), visits, fps=5, quality=8)

    rgb_path = "./logs/blender_paper_lego/renderonly_test_349999_bak"
    rgbs = []

    for i in range(25):
        img = Image.open(os.path.join(rgb_path, '{:03d}_lighting.png'.format(i)))
        rgbs.append(np.array(img))


    rgbs = np.stack(rgbs, 0)
    imageio.mimwrite(os.path.join("./videos", 'RGBS_original.mp4'), rgbs, fps=5, quality=8)