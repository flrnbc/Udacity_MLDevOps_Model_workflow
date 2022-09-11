#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
from configparser import MAX_INTERPOLATION_DEPTH
import logging
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args) # TODO: what exactly is updated?

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    
    logger.info("Downloading input artifact.") 
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    # store artifact in dataframe
    df = pd.read_csv(artifact_local_path)

    logger.info("Data cleaning: only keep apartments with price in the range [min_price, max_price].")
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()

    logger.info("Data cleaning: revert last_review to datetime.")
    df["last_review"] = pd.to_datetime(df["last_review"])

    logger.info("Impose geolocation restriction.")
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    logger.info("Save and log output artifact.")
    # export to csv
    df.to_csv("clean_sample.csv", index=False) # TODO: index?
    # save to output artifact
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    ) 
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This steps cleans the data")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name of the input artifact.",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of output artifact.",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact.",
        required=True
    )
    
    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the output artifact.",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price of apartments to keep.",
        required=True
    )
    
    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price of apartments to keep.",
        required=True
    )

    args = parser.parse_args()

    go(args)
