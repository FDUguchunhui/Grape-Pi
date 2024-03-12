'''
Script used for processing csv file from water
'''
import sys
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Pre-processing Water Mass Spectrometry data',
                                 epilog='This is a command line program designed for pre-processing data from Water Mass'
                                        'spectrometry instrument with a desired format that can be readily used for '
                                        'Graph neural network training.')

parser.add_argument('protein_data', metavar='protein-data', type=str,
                    help='the path to the intermediate processed file from Water instruments, should be in csv format')

parser.add_argument('-a', '--additional-protein-data', type=str,
                    help='the path to the additional protein data file contains additional protein features information and'
                         'could also be used for filtering the proteins in the proteinData that must be in the '
                         'additionalProteinData. Typically, this file is downloaded from public protein database,'
                         'e.g. neXtProt, UniProt')

parser.add_argument('-i', '--identifier-protein-data', type=str, default='protein.Accession',
                    help='a string for indicating the unique identifier in the proteinData file')

parser.add_argument('-d', '--identifier-additional-protein-data', type=str, default='Entry',
                    help='a string for indicating the unique identifier in the additionalProteinData file')

parser.add_argument('-o', '--output-file-name', type=str,
                    help='the path to save the output file. If not specified, by default is current working directory.'
                         'the default name is the proteinData file name + "_processed". ')


# parse command line argument
args = parser.parse_args()
protein_data_id = args.identifier_protein_data
additional_data_id = args.identifier_additional_protein_data

# import data
protein_data = pd.read_csv(args.protein_data)

if args.additional_protein_data is not None:
    additional_protein_data = pd.read_table(args.additional_protein_data, sep='\t')
    # extract unique proteins in the additional protein data
    proteins_in_additional_data = additional_protein_data[additional_data_id]
    # only keep proteins that are also in the additional data
    protein_data = protein_data[protein_data[protein_data_id].isin(proteins_in_additional_data)]

# for duplicate proteins in the protein data, only take the item that has largest number of peptides
protein_data = protein_data.sort_values('protein.matchedPeptides').groupby(protein_data_id).nth(0)

if args.output_file_name is not None:
    protein_data.to_csv(args.output_file_name)
else:
    path = args.protein_data + '_process.csv'
    protein_data.to_csv(path)



