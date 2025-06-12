from abc import ABC, abstractmethod


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import figure


class MissingValuesAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        self.identify_missing_values(df)
        self.visualize_missing_values(df)


    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame):
        pass


class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame):
        print("\nMissing Values Count by Columns:")
        missing_values = df.isnull().sum()
        print(missing_values)

    def visualize_missing_values(self, df: pd.DataFrame):
        print("\nVisualizing Missing Values...")
        plt.figure(figure=(12, 8))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.show()
