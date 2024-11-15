import argparse
import os
from tabulate import tabulate
from data_utils.face_detection import *
from VideoCop_Detector.utils import *
from VideoCop_Detector.DeepFakeDetectionModel import *
import torchvision
from data_utils.datasets import *
import warnings
import multiprocessing
import sys

warnings.filterwarnings("ignore", category=DeprecationWarning)



def predict_deepfake(input_videofile, df_method, debug=False, verbose=False):
    num_workers = multiprocessing.cpu_count() - 2
    model_params = dict()
    model_params['batch_size'] = 32
    model_params['imsize'] = 224
    model_params['encoder_name'] = 'tf_efficientnet_b0_ns'

    prob_threshold_fake = 0.5
    fake_fraction = 0.5

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    vid = os.path.basename(input_videofile)[:-4]
    output_path = os.path.join("output", vid)
    plain_faces_data_path = os.path.join(output_path, "plain_frames")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(plain_faces_data_path, exist_ok=True)

    if verbose:
        print(f'Extracting faces from the video')
    extract_landmarks_from_video(input_videofile, output_path, overwrite=True)
    crop_faces_from_video(input_videofile, output_path, plain_faces_data_path, overwrite=True)

    if df_method == 'plain_frames':
        model_path = 'final.chkpt'
        frames_path = plain_faces_data_path
    else:
        raise Exception("Unknown method")

    if verbose:
        print(f'Detecting DeepFakes using method: {df_method}')
    model = DeepFakeDetectionModel(frame_dim=model_params['imsize'], encoder_name=model_params['encoder_name'])
    if verbose:
        print(f'Loading model weights {model_path}')
    check_point_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(check_point_dict['model_state_dict'])

    model = model.to(device)
    model.eval()

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((model_params['imsize'], model_params['imsize'])),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
    ])

    data_path = os.path.join(frames_path, vid)
    test_dataset = SimpleImageFolder(root=data_path, transforms_=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=model_params['batch_size'], num_workers=num_workers,
                             pin_memory=True)
    if len(test_loader) == 0:
        print('Cannot extract images. Dataloaders empty')
        return None, None, None
    probabilities = []
    all_filenames = []
    all_predicted_labels = []
    with torch.no_grad():
        for batch_id, samples in enumerate(test_loader):
            frames = samples[0].to(device)
            output = model(frames)
            predicted = get_predictions(output).to('cpu').detach().numpy()
            class_probability = get_probability(output).to('cpu').detach().numpy()
            if len(predicted) > 1:
                all_predicted_labels.extend(predicted.squeeze())
                probabilities.extend(class_probability.squeeze())
                all_filenames.extend(samples[1])
            else:
                all_predicted_labels.append(predicted.squeeze())
                probabilities.append(class_probability.squeeze())
                all_filenames.append(samples[1])

        total_number_frames = len(probabilities)
        probabilities = np.array(probabilities)

        fake_frames_high_prob = probabilities[probabilities >= prob_threshold_fake]
        number_fake_frames = len(fake_frames_high_prob)
        if number_fake_frames == 0:
            fake_prob = 0
        else:
            fake_prob = round(sum(fake_frames_high_prob) / number_fake_frames, 4)

        real_frames_high_prob = probabilities[probabilities < prob_threshold_fake]
        number_real_frames = len(real_frames_high_prob)
        if number_real_frames == 0:
            real_prob = 0
        else:
            real_prob = 1 - round(sum(real_frames_high_prob) / number_real_frames, 4)

        pred = pred_strategy(number_fake_frames, number_real_frames, total_number_frames,
                             fake_fraction=fake_fraction)

        if debug:
            print(f'all {probabilities}')
            print(f'real {real_frames_high_prob}')
            print(f'fake {fake_frames_high_prob}')
            print(
                f"number_fake_frames={number_fake_frames}, number_real_frames={number_real_frames}, total_number_frames={total_number_frames}, fake_fraction={fake_fraction}")
            print(f'fake_prob = {round(fake_prob * 100, 4)}%, real_prob = {round(real_prob * 100, 4)}%  pred={pred}')
        return fake_prob, real_prob, pred


def individual_test():
    print_line()
    debug = False
    verbose = True
    fake_prob, real_prob, pred = predict_deepfake(args.input_videofile, args.method, debug=debug, verbose=verbose)
    if pred is None:
        print_red('Failed to detect DeepFakes')
        return

    label = "REAL" if pred == 0 else "DEEP-FAKE"

    probability = real_prob if pred == 0 else fake_prob
    probability = round(probability * 100, 4)
    print_line()
    if pred == 0:
        print_green(f'The video is {label}, probability={probability}%')
    else:
        print_red(f'The video is {label}, probability={probability}%')
    print_line()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DeepFakes detection App. \n Use demo mode or provide input_videofile and method')
    parser.add_argument('--input_videofile', action='store', help='Input video file')
    parser.add_argument('--method', action='store', choices=['plain_frames'],
                        help='Method type')
    args = parser.parse_args()
    if args.input_videofile is not None:
        if args.method is None:
            parser.print_help(sys.stderr)
        else:
            if os.path.isfile(args.input_videofile):
                individual_test()
            else:
                print(f'input file not found ({args.input_videofile})')
    else:
        parser.print_help(sys.stderr)
