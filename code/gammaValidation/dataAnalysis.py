import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime
import numpy as np
from typing import List, Dict

# File paths in camelCase
validationDataPath = "data/AVeriTecData/dataDev.json"
trainDataPath = "data/AVeriTecData/dataTrain.json"
testDataPath = "data/AVeriTecData/dataTest.json"
savePath = "docs/dataAnalysis"

def loadData() -> List[Dict]:
    """Load JSON data from validation dataset"""
    with open(validationDataPath, 'r') as file:
        data = json.load(file)
    return data

def loadAndPrepareData(filePath: str) -> pd.DataFrame:
    """Load JSON data and convert it to a pandas DataFrame"""
    with open(filePath, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data)

def analyzeClaimLabels(df: pd.DataFrame) -> Dict:
    """Analyze the distribution of claim labels"""
    labelCounts = df['label'].value_counts()
    return {
        'counts': labelCounts.to_dict(),
        'percentages': (labelCounts / len(df) * 100).round(2).to_dict()
    }

def analyzeReportingSources(df: pd.DataFrame) -> Dict:
    """Analyze the distribution of reporting sources"""
    sourceCounts = df['reporting_source'].value_counts()
    return {
        'counts': sourceCounts.to_dict(),
        'percentages': (sourceCounts / len(df) * 100).round(2).to_dict()
    }

def analyzeClaimTypes(df: pd.DataFrame) -> Dict:
    """Analyze the distribution of claim types"""
    allTypes = [ct for types in df['claim_types'].dropna() for ct in types]
    typeCounts = Counter(allTypes)
    total = sum(typeCounts.values())
    return {
        'counts': dict(typeCounts),
        'percentages': {k: round(v / total * 100, 2) for k, v in typeCounts.items()}
    }

def createVisualizations(df: pd.DataFrame) -> None:
    """Create and save various visualizations to the savePath folder"""
    # Set basic style and apply seaborn styling
    plt.style.use('default')
    sns.set_theme(style="whitegrid")
    
    # 1. Label Distribution Pie Chart
    plt.figure(figsize=(10, 6))
    labelAnalysis = analyzeClaimLabels(df)
    colors = sns.color_palette("husl", len(labelAnalysis['percentages']))
    plt.pie(list(labelAnalysis['percentages'].values()),
            labels=list(labelAnalysis['percentages'].keys()),
            autopct='%1.1f%%',
            colors=colors)
    plt.title('Distribution of Claim Labels', pad=20)
    plt.savefig(f"{savePath}/labelDistributionValidationData.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. Reporting Sources Bar Chart
    plt.figure(figsize=(12, 6))
    sourceAnalysis = analyzeReportingSources(df)
    sources = list(sourceAnalysis['counts'].keys())
    counts = list(sourceAnalysis['counts'].values())
    sns.barplot(x=sources, y=counts, palette="deep")
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribution of Reporting Sources', pad=20)
    plt.xlabel('Source')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f"{savePath}/sourceDistributionValidationData.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    # 3. Claim Types Distribution Bar Chart
    plt.figure(figsize=(12, 6))
    typeAnalysis = analyzeClaimTypes(df)
    types = list(typeAnalysis['counts'].keys())
    typeCounts = list(typeAnalysis['counts'].values())
    sns.barplot(y=types, x=typeCounts, palette="deep", orient='h')
    plt.title('Distribution of Claim Types', pad=20)
    plt.xlabel('Count')
    plt.tight_layout()
    plt.savefig(f"{savePath}/claimTypeValidationData.png", bbox_inches='tight', dpi=300)
    plt.close()

def createMermaidDiagram(df: pd.DataFrame) -> str:
    """Create a Mermaid flowchart showing the fact-checking process using data from the DataFrame"""
    supportedCount = len(df[df['label'] == 'Supported'])
    refutedCount = len(df[df['label'] == 'Refuted'])
    factCheckingStrategiesCount = len(df[df['fact_checking_strategies'].apply(lambda x: 'Written Evidence' in x if isinstance(x, list) else False)])
    otherStrategiesCount = len(df[df['fact_checking_strategies'].apply(lambda x: 'Written Evidence' not in x if isinstance(x, list) else True)])
    totalClaims = len(df)
    
    return f"""
    graph TD
        A[Total Claims: {totalClaims}] --> B[Fact Checking Process]
        B --> C[Supported: {supportedCount}]
        B --> D[Refuted: {refutedCount}]
        C --> E[Fact Checking Strategies]
        D --> E
        E --> F[Written Evidence: {factCheckingStrategiesCount}]
        E --> G[Other Strategies: {otherStrategiesCount}]
    """

def main():
    try:
        # Load data
        print("Loading data...")
        data = loadData()
        
        # Convert to DataFrame
        print("Converting to DataFrame...")
        df = pd.DataFrame(data)
        
        # Generate visualizations into the savePath folder
        print("Creating visualizations...")
        createVisualizations(df)
        
        # Create Mermaid diagram
        print("Generating Mermaid diagram...")
        mermaidDiagram = createMermaidDiagram(df)
        
        # Print analysis results
        print("\nAnalysis Results:")
        print("\nLabel Distribution:")
        print(analyzeClaimLabels(df))
        
        print("\nReporting Sources Distribution:")
        print(analyzeReportingSources(df))
        
        print("\nClaim Types Distribution:")
        print(analyzeClaimTypes(df))
        
        print("\nMermaid Diagram:")
        print(mermaidDiagram)
        
        print("\nVisualization files saved in:", savePath)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find the data file at {validationDataPath}")
        print("Please check if the file path is correct.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
