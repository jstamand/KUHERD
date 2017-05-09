#    Copyright (C) 2017  Joseph St.Amand

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..",".."))

import argparse
import pandas as pd
import pickle
import yaml

from KUHERD.Models import PurposeFieldModel
from KUHERD.LabelTransformations import vec2mat, label2mat, mat2vec


def main():
    parser = argparse.ArgumentParser(description='Utility to make predictions on new data')
    parser.add_argument('--data', dest='data', type=str, help='XLSX file containing input data')
    parser.add_argument('--config', dest='config', type=str, help='Input configuration file')
    parser.add_argument('--model', dest='model', type=str, help='name of the model file to create')
    args = parser.parse_args()

    # check that all argument are available
    if args.model is None:
        print('--model option required!\n')
        exit()
    elif args.data is None:
        print('--data option required!\n')
        exit()
    elif args.config is None:
        print('--config option required!\n')
        exit()

    # check that the input data given exists
    if not os.path.isfile(args.data):
        print('Given input data file: %s does not exist!\n' % args.data)
        exit()

    # check that the input config given exists
    if not os.path.isfile(args.config):
        print('Given input config file: %s does not exist!\n' % args.config)
        exit()


    # load the configuration and initialize the model
    config = None

    with open(args.config, 'r') as fin:
        config = yaml.safe_load(fin)
    model = PurposeFieldModel(config)

    # load the data
    df = pd.read_excel(args.data)

    # verify that the required fields are there
    if 'sow'not in df.keys():
        sys.exit('Expecting column named \'sow\' in given excel file!\n')
    elif 'purpose' not in df.keys():
        sys.exit('Expecting column named \'purpose\' in given excel file!\n')
    elif 'field' not in df.keys():
        sys.exit('Expecting column named \'field\' in given excel file!\n')

    # check that all the labels given are valid and convert to matrix form
    Y_purpose = label2mat(list(df['purpose']), 'purpose')
    Y_field = label2mat(list(df['field']), 'field')

    abstracts = list(df['sow'])

    # use the data to fit the model
    model.fit(abstracts, Y_purpose, Y_field)

    # pickle the model and save it
    with open(args.model, 'wb') as fout:
        pickle.dump(model, fout, protocol=pickle.HIGHEST_PROTOCOL)

    exit()


if __name__ == "__main__":
    main()
