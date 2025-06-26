import os
import zipfile
from abc import ABC, abstractmethod

import pandas as pd


class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path:str) -> pd.DataFrame:
        pass


class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path:str) -> pd.DataFrame:

        if not file_path.endswith(".zip"):
            raise ValueError("File extension must be .zip")

        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall("extracted_data")

        extracted_files = os.listdir("extracted_data")
        csv_files = [f for f in extracted_files if f.endswith(".csv")]

        if len(csv_files) == 0:
            raise ValueError("No .csv files found in extracted_data")
        if len(csv_files) > 1:
            raise ValueError("More than one .csv files found in extracted_data")

        csv_file_path = os.path.join("extracted_data", csv_files[0])
        df = pd.read_csv(csv_file_path)

        return df

class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        """Returns the appropriate DataIngestor based on the file extension."""
        if file_extension == ".zip":
            return ZipDataIngestor()
        else:
            raise ValueError(f"No ingestor available for file extension {file_extension}")