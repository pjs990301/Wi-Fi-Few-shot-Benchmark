import importlib

from scapy.all import *
import dataloader.config as config
import pandas as pd
import numpy as np

decoder = importlib.import_module(f'dataloader.decoders.{config.decoder}') # This is also an import


def pcap_to_df(filename, bandwidth=80, amp=True, del_null=True):
    nulls = {
        20: [x + 32 for x in [
            -32, -31, -30, -29,
            31, 30, 29, 0
        ]],

        40: [x + 64 for x in [
            -64, -63, -62, -61, -60, -59, -1,
            63, 62, 61, 60, 59, 1, 0
        ]],

        80: [x + 128 for x in [
            -128, -127, -126, -125, -124, -123, -1,
            127, 126, 125, 124, 123, 1, 0
        ]],

        160: [x + 256 for x in [
            -256, -255, -254, -253, -252, -251, -129, -128, -127, -5, -4, -3, -2, -1,
            255, 254, 253, 252, 251, 129, 128, 127, 5, 4, 3, 3, 1, 0
        ]]
    }

    # Read pcap file and create dataframe
    try:
        csi_samples = decoder.read_pcap(filename)
    except FileNotFoundError:
        print(f'File {filename} not found.')
        exit(-1)

    # Create dataloader data frame
    if amp is True:
        csi_df = pd.DataFrame(np.abs(csi_samples.get_all_csi()))  # Get dataloader amplitude dataframe
    else:
        csi_df = pd.DataFrame(csi_samples.get_all_csi())  # Get I/Q complex num dataframe

    if del_null is True:
        csi_df = csi_df[csi_df.columns.difference(nulls[bandwidth])]

    return csi_df


if __name__ == "__main__":
    filename = "../few_shot_datasets/pcap/Empty_Ex_Home_13.pcap"
    df = pcap_to_df(filename)

    print(df.iloc[:242])
    print(np.array(df).shape)
