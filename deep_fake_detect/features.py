from glob import glob
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import multiprocessing
from tqdm import tqdm
import pandas as pd
from data_utils.utils import *


def generate_frame_label_csv(mode=None, dataset=None):
    if mode == 'train':
        originals_, fakes_ = get_training_reals_and_fakes()
        if dataset == 'plain':
            csv_file = ConfigParser.getInstance().get_dfdc_train_frame_label_csv_path()
            crop_path = ConfigParser.getInstance().get_dfdc_crops_train_path()
        else:
            raise Exception('Bad dataset')
    elif mode == 'valid':
        originals_, fakes_ = get_valid_reals_and_fakes()
        if dataset == 'plain':
            csv_file = ConfigParser.getInstance().get_dfdc_valid_frame_label_csv_path()
            crop_path = ConfigParser.getInstance().get_dfdc_crops_valid_path()
        else:
            raise Exception('Bad dataset')

    elif mode == 'test':
        originals_, fakes_ = get_test_reals_and_fakes()
        if dataset == 'plain':
            csv_file = ConfigParser.getInstance().get_dfdc_test_frame_label_csv_path()
            crop_path = ConfigParser.getInstance().get_dfdc_crops_test_path()
        else:
            raise Exception('Bad dataset')
    else:
        raise Exception('Bad mode in generate_frame_label_csv')

    originals = [os.path.splitext(video_filename)[0] for video_filename in originals_]
    fakes = [os.path.splitext(video_filename)[0] for video_filename in fakes_]

    print(f'mode {mode}, csv file : {csv_file}')
    df = pd.DataFrame(columns=['video_id', 'frame', 'label'])

    crop_ids = glob(crop_path + '/*')
    results = []
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        for cid in tqdm(crop_ids, desc='Scheduling jobs to label frames'):
            jobs.append(pool.apply_async(get_video_frame_labels_mapping, (cid, originals, fakes,)))

        for job in tqdm(jobs, desc="Labeling frames"):
            r = job.get()
            results.append(r)

    for r in tqdm(results, desc='Consolidating results'):
        df = df.append(r, ignore_index=True)
    df.set_index('video_id', inplace=True)
    df.to_csv(csv_file)


def generate_frame_label_csv_files():
    modes = ['train', 'valid', 'test']
    datasets = ['plain']
    for d in datasets:
        print(f'Generating frame_label csv for dataset {d}')
        for m in modes:
            print(f'Generating frame_label csv for processed {m} samples')
            generate_frame_label_csv(mode=m, dataset=d)


def generate_DFDC_MRIs():
    print('MRI related functions are removed from this script')


def predict_mri_using_MRI_GAN(crops_path, mri_path, vid, imsize, overwrite=False):
    print('MRI related functions are removed from this script')


def predict_mri_using_MRI_GAN_batch(crops_path, mri_path):
    print('MRI related functions are removed from this script')
