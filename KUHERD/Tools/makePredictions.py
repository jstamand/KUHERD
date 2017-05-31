import argparse
import os
import pandas as pd
import pickle
import sys



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
    Y_purpose_str = models.purpose_model.labelset.vec2string(predictions['purpose'])
    Y_field_str = models.field_model.labelset.vec2string(predictions['field'])

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