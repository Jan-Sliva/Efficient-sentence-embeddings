import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import os.path as P
import numpy as np
from adjustText import adjust_text

def load_data(csv_path):
    return pd.read_csv(csv_path)

def load_config(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def create_graph(data, config, output_path):
    # Increase the default font size
    plt.rcParams.update({'font.size': 24})
    
    for graph in config:
        plt.figure(figsize=(14, 8))  # Increased width to accommodate legend
        
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
        
        x = list(combine_values(x_config['columns'], x_config['function']))
        y = list(combine_values(y_config['columns'], y_config['function']))
        labels = list(data[graph['label']])
        
        # Create a color map for unique labels
        unique_labels = list(set(labels))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        label_to_color = dict(zip(unique_labels, colors))
        
        # Plot points with different colors
        for label in unique_labels:
            mask = [l == label for l in labels]
            x_points = [x[i] for i, m in enumerate(mask) if m]
            y_points = [y[i] for i, m in enumerate(mask) if m]
            plt.scatter(x_points, y_points, 
                       marker='.', 
                       s=200, 
                       color=label_to_color[label],
                       label=label)
        
        plt.xlabel(graph['x_axis']['label'], fontsize=24)
        plt.ylabel(graph['y_axis']['label'], fontsize=24)
        plt.title(graph['title'], fontsize=28)
        plt.grid(True)
        
        # Add legend outside of the plot
        plt.legend(bbox_to_anchor=(1.05, 1), 
                  loc='upper left', 
                  fontsize=12,
                  borderaxespad=0.)
        
        # Set both axes to start at zero
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        
        plt.tight_layout()  # Adjust layout to prevent legend cutoff
        
        # Increase font size for tick labels
        plt.tick_params(axis='both', which='major', labelsize=20)
        
        plt.savefig(P.join(output_path, f"{graph['save_name']}.png"), 
                    dpi=300,
                    bbox_inches='tight')  # Added bbox_inches to prevent legend cutoff
    
    # Reset the font size to default after creating all graphs
    plt.rcParams.update({'font.size': plt.rcParamsDefault['font.size']})

def main():
    parser = argparse.ArgumentParser(description='Generate graph from CSV data based on JSON configuration.')
    parser.add_argument('--csv_path', type=str, help='Path to the CSV file containing the data')
    parser.add_argument('--json_path', type=str, help='Path to the JSON file containing graph configurations')
    parser.add_argument('--output_path', type=str, help='Path to the output folder')
    args = parser.parse_args()

    data = load_data(args.csv_path)
    config = load_config(args.json_path)

    data = data.query('hide != True')

    create_graph(data, config, args.output_path)

if __name__ == "__main__":
    main()
