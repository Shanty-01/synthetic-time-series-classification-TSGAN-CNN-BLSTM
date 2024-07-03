import sys
import argparse
import os
import sys
import glob
import tempfile
import numpy as np

import keras
from scipy.linalg import sqrtm
from sp_tool.arff_helper import ArffHelper
import sp_tool.util as sp_util
from sp_tool.evaluate import CORRESPONDENCE_TO_HAND_LABELLING_VALUES
from sp_tool import recording_processor as sp_processor

import blstm_model
from blstm_model import zip_equal

def run(args):
    """
    Run prediction for a trained model on a set of .arff files (with features already extracted).
    See feature_extraction folder for the code to compute appropriate features.
    :param args: command line arguments
    :return: a list of tuples (corresponding to all processed files) that consist of
               - the path to an outputted file
               - predicted per-class probabilities
    """
    subfolders_and_fnames = find_all_subfolder_prefixes_and_input_files(args)
    out_fnames = get_corresponding_output_paths(subfolders_and_fnames, args)

    print('Processing {} file(s) from "{}" into "{}"'.format(len(out_fnames),
                                                             args.input,
                                                             args.output))

    arff_objects = [ArffHelper.load(open(fname)) for _, fname in subfolders_and_fnames]

    keys_to_keep = blstm_model.get_arff_attributes_to_keep(args)
    print('Will look for the following keys in all .arff files: {}. ' \
          'If any of these are missing, an error will follow!'.format(keys_to_keep))
    all_features = [get_features_columns(obj, args) for obj in arff_objects]
    print(f'the length of input is : {len(all_features)}')
    print(f'the length of the input 2 is : {len(all_features[0])}')
    print(f'the length of the input 3 is : {len(all_features[0][0])}')

    model = keras.models.load_model(args.model_path,
                                    custom_objects={'f1_SP': blstm_model.f1_SP,
                                                    'f1_SACC': blstm_model.f1_SACC,
                                                    'f1_FIX': blstm_model.f1_FIX})
    
    # Guess the padding size from model input and output size
    window_length = model.output_shape[1]  # (batch size, window size, number of classes)
    padded_window_shape = model.input_shape[1]  # (batch size, window size, number of features)
    padding_features = (padded_window_shape - window_length) // 2
    print('Will pad the feature sequences with {} samples on each side.'.format(padding_features))

    keys_to_subtract_start = sorted({'x', 'y'}.intersection(keys_to_keep))
    if len(keys_to_subtract_start) > 0:
        print('Will subtract the starting values of the following features:', keys_to_subtract_start)
    keys_to_subtract_start_indices = [i for i, key in enumerate(keys_to_keep) if key in keys_to_subtract_start]
    prediction = blstm_model.get_extracted_feature(model=model,
                                                X=all_features,
                                                y=None,  # no ground truth available or needed
                                                keys_to_subtract_start_indices=keys_to_subtract_start_indices,
                                                correct_for_unknown_class=False,
                                                padding_features=padding_features,
                                                split_by_items=True)
        
    print(f'prediction 1 : {prediction.shape}')

#######################################################################################

    subfolders_and_fnames = find_all_subfolder_prefixes_and_input_files_synth(args)
    out_fnames = get_corresponding_output_paths(subfolders_and_fnames, args)

    print('Processing {} file(s) from "{}" into "{}"'.format(len(out_fnames),
                                                             args.input_synth,
                                                             args.output))

    arff_objects = [ArffHelper.load(open(fname)) for _, fname in subfolders_and_fnames]

    keys_to_keep = blstm_model.get_arff_attributes_to_keep(args)
    print('Will look for the following keys in all .arff files: {}. ' \
          'If any of these are missing, an error will follow!'.format(keys_to_keep))
    all_features = [get_features_columns(obj, args) for obj in arff_objects]
    print(f'the length of input is : {len(all_features)}')
    print(f'the length of the input 2 is : {len(all_features[0])}')
    print(f'the length of the input 3 is : {len(all_features[0][0])}')
    model = keras.models.load_model(args.model_path,
                                    custom_objects={'f1_SP': blstm_model.f1_SP,
                                                    'f1_SACC': blstm_model.f1_SACC,
                                                    'f1_FIX': blstm_model.f1_FIX})
    
    # Guess the padding size from model input and output size
    window_length = model.output_shape[1]  # (batch size, window size, number of classes)
    padded_window_shape = model.input_shape[1]  # (batch size, window size, number of features)
    padding_features = (padded_window_shape - window_length) // 2
    print('Will pad the feature sequences with {} samples on each side.'.format(padding_features))

    keys_to_subtract_start = sorted({'x', 'y'}.intersection(keys_to_keep))
    if len(keys_to_subtract_start) > 0:
        print('Will subtract the starting values of the following features:', keys_to_subtract_start)
    keys_to_subtract_start_indices = [i for i, key in enumerate(keys_to_keep) if key in keys_to_subtract_start]
    np_allfeature = np.concatenate(all_features)
    np_allfeature = np_allfeature.reshape(-1, 5020, 2)
    pred_list_hat = []
    for i in np_allfeature :
        all_feature = [i]
        print(f'the length of all_feature is {len(all_feature)}')
        predictions_hat = blstm_model.get_extracted_feature(model=model,
                                                X=all_feature,
                                                y=None,  # no ground truth available or needed
                                                keys_to_subtract_start_indices=keys_to_subtract_start_indices,
                                                correct_for_unknown_class=False,
                                                padding_features=padding_features,
                                                split_by_items=True)
        pred_list_hat.append(predictions_hat)
    print(f'prediction 2 : {len(pred_list_hat), len(pred_list_hat[0]), len(pred_list_hat[0][0]), len(pred_list_hat[0][0][0])}')
    prediction = prediction.reshape((prediction.shape[0]*prediction.shape[1],prediction.shape[2]))
    fid_scores = []
    for i in pred_list_hat:
        prediction_hat = i.reshape((i.shape[0]*i.shape[1],i.shape[2]))
        fid_score = calculate_fid(prediction, prediction_hat)
        print(f'fid score : {fid_score}')
        fid_scores.append(fid_score)
    fid_score_same = calculate_fid(prediction, prediction)
    avg_fid = sum(fid_scores) / len(fid_scores)
    print(f'FID between original files : {fid_score_same}')
    print(f'Average of all FID scores between original and synthetic data : {avg_fid}')

# calculate frechet inception distance
def calculate_fid(pred, pred_hat):
    # calculate mean and covariance statistics
    mu1, sigma1 = pred.mean(axis=0), np.cov(pred, rowvar=False)
    mu2, sigma2 = pred_hat.mean(axis=0), np.cov(pred_hat, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def find_all_subfolder_prefixes_and_input_files(args):
    """
    Extract a matching set of paths to .arff files and additional folders between the --input folder and the files
    themselves (so that we will be able to replicate the structure later on)
    :param args: command line arguments
    :return: a list of tuples, where the first element is the sub-folder prefix and the second one is the full path
             to each .arff file
    """
    if os.path.isfile(args.input):
        return [('', args.input)]
    assert os.path.isdir(args.input), '--input is neither a file nor a folder'

    res = []
    for dirpath, dirnames, filenames in os.walk(args.input):
        filenames = [x for x in filenames if x.lower().endswith('.arff')]
        if filenames:
            dirpath_suffix = dirpath[len(args.input):].strip('/')
            res += [(dirpath_suffix, os.path.join(dirpath, fname)) for fname in filenames]
    return res

def find_all_subfolder_prefixes_and_input_files_synth(args):
    """
    Extract a matching set of paths to .arff files and additional folders between the --input folder and the files
    themselves (so that we will be able to replicate the structure later on)
    :param args: command line arguments
    :return: a list of tuples, where the first element is the sub-folder prefix and the second one is the full path
             to each .arff file
    """
    if os.path.isfile(args.input_synth):
        return [('', args.input_synth)]
    assert os.path.isdir(args.input_synth), '--input_synth is neither a file nor a folder'

    res = []
    for dirpath, dirnames, filenames in os.walk(args.input_synth):
        filenames = [x for x in filenames if x.lower().endswith('.arff')]
        if filenames:
            dirpath_suffix = dirpath[len(args.input_synth):].strip('/')
            res += [(dirpath_suffix, os.path.join(dirpath, fname)) for fname in filenames]
    return res

def get_corresponding_output_paths(subfolders_and_full_input_filenames, args):
    """
    Create a list that will contain output paths for all the @subfolders_and_full_input_filenames
    (the output of find_all_subfolder_prefixes_and_input_files() function) in the output folder.
    :param subfolders_and_full_input_filenames: subfolder prefixes,
           returned by find_all_subfolder_prefixes_and_input_files()
    :param args: command line arguments
    :return:
    """
    if args.output is None:
        args.output = tempfile.mkdtemp(prefix='blstm_model_output_')
        print('No --output was provided, creating a folder in', args.output, file=sys.stderr)

    if args.output.lower().endswith('.arff'):
        assert len(subfolders_and_full_input_filenames) == 1, 'If --output is just one file, cannot have more than ' \
                                                              'one input file! Consider providing a folder as the ' \
                                                              '--output.'
        return [args.output]

    res = []
    for subfolder, full_name in subfolders_and_full_input_filenames:
        res.append(os.path.join(args.output, subfolder, os.path.split(full_name)[-1]))
    return res

def get_features_columns(arff_obj, args):
    """
    Extracting features from the .arff file (reading the file, getting the relevant columns
    :param arff_obj: a loaded .arff file
    :param args: command line arguments
    :return:
    """
    keys_to_keep = blstm_model.get_arff_attributes_to_keep(args)

    keys_to_convert_to_degrees = ['x', 'y'] + [k for k in keys_to_keep if 'speed_' in k or 'acceleration_' in k]
    keys_to_convert_to_degrees = sorted(set(keys_to_convert_to_degrees).intersection(keys_to_keep))
    # Conversion is carried out by dividing by pixels-per-degree value (PPD)
    if get_features_columns.run_count == 0:
        if len(keys_to_convert_to_degrees) > 0:
            print('Will divide by PPD the following features', keys_to_convert_to_degrees)
    get_features_columns.run_count += 1

    # normalize coordinates in @o by dividing by @ppd_f -- the pixels-per-degree value of the @arff_obj
    ppd_f = sp_util.calculate_ppd(arff_obj)
    for k in keys_to_convert_to_degrees:
        arff_obj['data'][k] /= ppd_f

    # add to respective data sets (only the features to be used and the true labels)
    return np.hstack([np.reshape(arff_obj['data'][key], (-1, 1)) for key in keys_to_keep]).astype(np.float64)

def parse_args():
    # Will keep most of the arguments, but suppress others
    base_parser = blstm_model.parse_args(dry_run=True)
    # Inherit all arguments, but retain the possibility to add the same args, but suppress them
    parser = argparse.ArgumentParser(parents=[base_parser], add_help=False, conflict_handler='resolve')

    # List all arguments (as lists of all ways to address each) that are to be eradicated
    args_to_suppress = [
        ['--model-name', '--model'],  # will add a more intuitive --model-path argument below
        # no need for the following when training is completed already
        ['--initial-epoch'],
        ['--batch-size'],
        ['--run-once', '--once', '-o'],
        ['--run-once-video'],
        ['--ground-truth-folder', '--gt-folder'],  # no need for ground truth
        ['--final-run', '--final', '-f'],  # it's always a "final" run here
        ['--folder', '--output-folder'],   # will override
        ['--training-samples'],
        ['--sp-tool-folder']
    ]

    for arg_group in args_to_suppress:
        parser.add_argument(*arg_group, help=argparse.SUPPRESS)

    parser.add_argument('--input', '--in', required=True,
                        help='Path to input data. Can be either a single .arff file, or a whole directory. '
                             'In the latter case, this directory will be scanned for .arff files, and all of them will '
                             'be used as inputs.')
    parser.add_argument('--input_synth', '--in_synth', required=True,
                        help='Path to input synth data. Can be either a single .arff file, or a whole directory. '
                             'In the latter case, this directory will be scanned for .arff files, and all of them will '
                             'be used as inputs.')

    # rewrite the help
    parser.add_argument('--output', '--output-folder', '--out', dest='output', default=None,
                        help='Write prediction results as ARFF file(s) here. Will mimic the structure of the --input '
                             'folder, or just create a single file, if --input itself points to an .arff file. '
                             'Can be a path to the desired output .arff file, in case --input is also just one file. '
                             'If not provided, will create a temporary folder and write the outputs there.')

    parser.add_argument('--model-path', '--model', default=None,
                        help='Path to a particular model (an .h5 file), which is to be used, or a folder containing '
                             'all 18 models that are trained in the Leave-One-Video-Out cross-validation procedure '
                             'on GazeCom. If this argument is '
                             'provided, it overrides all the architecture- and model-defining parameters. The '
                             'provided .h5 file will be loaded instead. \n\nIf --model-path is not provided, will '
                             'generate a model descriptor from architecture parameters and so on, and look for it '
                             'in the respective subfolder of ``data/models/''. Will then (or if --model-path contains '
                             'a path to a folder, and not to an .h5 file) take the model that was '
                             'trained on all data except for `bridge_1`, since this video has no "true" smooth '
                             'pursuit, so we will this way maximise the amount of this relatively rare class in the '
                             'used training set.')

    args = parser.parse_args()

    if args.model_path is None:
        model_descriptor = blstm_model.get_full_model_descriptor(args)
        args.model_path = 'data/models/LOO_{descr}/'.format(descr=model_descriptor)

    # If it is a path to a directory, find the model trained for the ``bridge_1'' clip.
    # Otherwise, we just assume that the path points to a model file.
    if os.path.isdir(args.model_path):
        all_model_candidates = sorted(glob.glob('{}/*_without_bridge_1*.h5'.format(args.model_path)))
        if len(all_model_candidates) == 0:
            raise ValueError('No model in the "{dir}" folder has ``without_bride_1\'\' in its name. Either pass '
                             'a path to an exact .h5 model file in --model-path, or make sure you have the right model '
                             'in the aforementioned folder.'.format(dir=args.model_path))
        elif len(all_model_candidates) > 1:
            raise ValueError('More than one model with ``without_bride_1\'\' in its name has been found in the "{dir}" '
                             'folder: {candidates}. Either pass a path to an exact .h5 model file in --model-path, '
                             'or make sure you have only one model trained without the clip ``bridge_1\'\' in the '
                             'aforementioned folder.'.format(dir=args.model_path,
                                                             candidates=all_model_candidates))
        args.model_path = all_model_candidates[0]  # since there has to be just one

    return args

get_features_columns.run_count = 0

if __name__ == '__main__':
    run(parse_args())
