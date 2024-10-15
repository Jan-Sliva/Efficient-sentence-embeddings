import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt

def load_data(csv_path):
    return pd.read_csv(csv_path)

def load_config(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def create_graph(data, config):
    plots = []
    
    for graph in config:
        plt.figure(figsize=(12, 8))
        
        x_config = graph['x_axis']
        y_config = graph['y_axis']
        
        def combine_values(columns, function):
            if function == 'sum':
                return sum([data[col] for col in columns])
            elif function == 'mean':
                values = [data[col] for col in columns]
                return sum(values) / len(values)
            else:
                raise ValueError(f"Unsupported function: {function}")
        
        x = combine_values(x_config['columns'], x_config['function'])
        y = combine_values(y_config['columns'], y_config['function'])
        labels = data[graph['label']]
        
        plt.scatter(x, y)
        
        for i, label in enumerate(labels):
            plt.annotate(label, (x[i], y[i]), xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel(graph['x_axis']['label'])
        plt.ylabel(graph['y_axis']['label'])
        plt.title(graph['title'])
        plt.grid(True)
        plt.tight_layout()
        
        plots.append(plt)
    
    return plots

def main():
    parser = argparse.ArgumentParser(description='Generate graph from CSV data based on JSON configuration.')
    parser.add_argument('--csv_path', type=str, help='Path to the CSV file containing the data')
    parser.add_argument('--json_path', type=str, help='Path to the JSON file containing graph configurations')
    args = parser.parse_args()

    data = load_data(args.csv_path)
    config = load_config(args.json_path)

    plots = create_graph(data, config)
    for plot in plots:
        plot.savefig(f"plot.png")

if __name__ == "__main__":
    main()
