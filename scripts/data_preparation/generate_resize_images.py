import cv2
import os
import sys
from os import path as osp
from multiprocessing import Pool
from glob import glob
from tqdm import tqdm


def main():
    """resize images

    Args:
        opt (dict): Configuration dict. It contains:
        n_thread (int): Thread number.
        compression_level (int):  CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and
            longer compression time. Use 0 for faster CPU decompression. Default: 3, same in cv2.
        input_folder (str): Path to the input folder.
        method (str): method for resize
        size (int): size of resize.
        save_folder (str): Path to save folder.

    Usage:
        For each folder, run this script.
        Typically, there are four folders to be processed for DIV2K dataset.

            * DIV2K_train_HR
            * DIV2K_train_LR_bicubic/X2
            * DIV2K_train_LR_bicubic/X3
            * DIV2K_train_LR_bicubic/X4

        After process, each sub_folder should have the same number of subimages.

        Remember to modify opt configurations according to your settings.
    """

    opt = {}
    opt['n_thread'] = 20
    opt['compression_level'] = 3

    # DIV2K HR images
    opt['input_folder'] = '/storage/datasets/DIV2K/DIV2K_train_depth_HR'
    opt['method'] = 'bicubic'
    opt['size'] = 4
    opt['save_folder'] = f"/storage/datasets/DIV2K/DIV2K_train_depth_LR_{opt['method']}/X{opt['size']}"
    resize_images(opt)

    # Set 5 or 14
    opt['input_folder'] = '/storage/datasets/Set5/Depth'
    opt['method'] = 'bicubic'
    opt['size'] = 4
    opt['save_folder'] = f"/storage/datasets/Set5/DepthLRbicx{opt['size']}"
    resize_images(opt)

    opt['input_folder'] = '/storage/datasets/Set14/Depth'
    opt['method'] = 'bicubic'
    opt['size'] = 4
    opt['save_folder'] = f"/storage/datasets/Set14/DepthLRbicx{opt['size']}"
    resize_images(opt)


def resize_images(opt):
    """resize images.

    Args:
        opt (dict): Configuration dict. It contains:
        input_folder (str): Path to the input folder.
        save_folder (str): Path to save folder.
        n_thread (int): Thread number.
    """
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists. Exit.')
        sys.exit(1)

    img_list = glob(f'{input_folder}/*.png')

    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(worker, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')


def worker(path, opt):
    """Worker for each process.

    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
        size (int): size of resize.
        save_folder (str): Path to save folder.
        compression_level (int): for cv2.IMWRITE_PNG_COMPRESSION.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    size = opt['size']
    img_name, extension = osp.splitext(osp.basename(path))

    # remove the x2, x3, x4 and x8 in the filename for DIV2K
    img_name = img_name.replace('-dpt_beit_large_512', '')

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    h, w = img.shape[0:2]

    # Define the new dimensions of the image
    n_h = int(h/size)
    n_w = int(w/size)


    # Resize the image using cv2.resize()
    resized_img = cv2.resize(img, (n_w, n_h), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(osp.join(opt['save_folder'], f'{img_name}{extension}'), resized_img, [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    process_info = f'Processing {img_name} ...'
    return process_info


if __name__ == '__main__':
    main()