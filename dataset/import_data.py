import argparse
import pathlib
import random
import os
from importers import *

description = """ Import filenames of dataset.
                  This script creates two textfiles, 'train.txt' and 'test.txt'.
                  The two files contain filenames of data.
                  Dataloaders read the files to acces files and loads them.
              """

def parse_args():
    parser = argparse.ArgumentParser(description = description)
    parser.add_argument("--data_type", dest="data_type", required=True, type=str, choices=['synth', 'real'])
    parser.add_argument("--data_dir", dest="data_dir", required=True, type=str, help="root directory of datset")
    parser.add_argument("--test_ratio", dest="test_ratio", default=0.2, type=float, help="ratio of test dataset size (only for synth data)")

    return parser.parse_args()

def write_filenames(filename, filenames_to_write):
    with open(filename, 'w') as f:
        for fn in filenames_to_write:
            line = ','.join(fn) + "\n"
            f.write(line)

def main(args):
    if args.data_type == 'synth':
        importer = SynthImporter()
    elif args.data_type == 'real':
        importer = RealImporter()
    else:
        raise ValueError('wrong data type')

    data_filenames_for_train, data_filenames_for_test = importer.get_file_names(args)

    this_file = pathlib.Path(os.path.abspath(__file__))
    this_dir = this_file.parents[0]
    import_root = this_dir.joinpath(importer.get_import_root())
    import_root.mkdir(exist_ok=True)

    train_filename = str(import_root.joinpath('train.txt'))
    write_filenames(train_filename, data_filenames_for_train)
    test_filename = str(import_root.joinpath('test.txt'))
    write_filenames(test_filename, data_filenames_for_test)

    print("number of train data: {}".format(len(data_filenames_for_train)))
    print("number of test data: {}".format(len(data_filenames_for_test)))

if __name__ == "__main__":
    args = parse_args()
    main(args)
