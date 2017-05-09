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

from KUHERD.LabelTransformations import vec2string

def main():

    parser = argparse.ArgumentParser(description='Utility to make predictions on new data')
    parser.add_argument('--model', dest='model', type=str, help='name of the model file')
    parser.add_argument('--data', dest='data', type=str, help='XLSX file containing input data')
    parser.add_argument('--output', dest='output', type=str, help='XLSX file to create as output')
    args = parser.parse_args()

    # check that all argument are available
    if args.model is None:
        print('--model option required!\n')
        exit()
    elif args.data is None:
        print('--data option required!\n')
        exit()
    elif args.output is None:
        print('--output option required!\n')
        exit()

    # check that both the model file and input data exist
    if not os.path.isfile(args.model):
        print('Given model file: %s does not exist!\n' % args.model)
        exit()
    elif not os.path.isfile(args.data):
        print('Given input data file: %s does not exist!\n' % args.data)
        exit()

    # load the data
    df = pd.read_excel(args.data)

    # verify that the required fields are there
    if 'sow' not in df.keys():
        sys.exit('Expecting column named \'sow\' in given excel file!\n')

    # load the model
    models = None
    with open(args.model, 'rb') as fin:
        models = pickle.load(fin)

    # load and verify the data
    df = pd.read_excel(args.data)

    if 'sow' not in df.keys():
        sys.exit('Expecting column named \'sow\' in given excel file!\n')

    # make predictions on the data
    abstracts = df['sow']
    predictions = models.predict(abstracts)

    # convert the label values into strings
    Y_purpose_str = vec2string(predictions['purpose'], 'purpose')
    Y_field_str = vec2string(predictions['field'], 'field')

    # place the predictions in a dataframe
    prediction_df = pd.DataFrame(data={'sow': abstracts, 'purpose': Y_purpose_str, 'field': Y_field_str}, columns=['sow', 'purpose', 'field'])

    # write the input dataframe to and excel file
    pd_writer = pd.ExcelWriter(args.output, engine='xlsxwriter')
    df.to_excel(pd_writer, sheet_name='Input')
    prediction_df.to_excel(pd_writer, sheet_name='Predictions')
    pd_writer.save()

    exit()


if __name__ == "__main__":
    main()
