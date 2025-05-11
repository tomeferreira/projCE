import os
import json
import re
import pandas as pd
import numpy as np


def extract_results_to_json_and_csv(txt_file_path, output_dir=None):
    """Convert a results.txt file to JSON and CSV formats."""
    # Initialize data dictionary
    data = {}

    # Initialize variables
    header_info = {}
    elapsed_time = None
    fitness_data = []

    with open(txt_file_path, 'r') as f:
        lines = f.readlines()

    # Parse header section
    i = 0
    while i < len(lines) and not lines[i].startswith('Generation |'):
        line = lines[i].strip()
        if line and ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            # Convert numeric values
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass

            header_info[key] = value

            # Extract elapsed time
            if 'Elapsed_Time' in line:
                match = re.search(r'(\d+\.\d+)', line)
                if match:
                    elapsed_time = float(match.group(1))
        i += 1

    # Skip header line of the table
    i += 1

    # Parse fitness data
    while i < len(lines):
        line = lines[i].strip()
        if line:
            parts = [part.strip() for part in line.split('|')]
            if len(parts) == 3 or (len(parts) == 1 and '|' not in line):
                # Handle both formats
                if len(parts) == 3:
                    gen, best_fit, avg_fit = parts
                else:
                    # For lines without '|' separators
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        gen, best_fit, avg_fit = parts[0], parts[1], parts[2]
                    else:
                        i += 1
                        continue

                try:
                    generation = int(gen)
                    best_fitness = float(best_fit)
                    avg_fitness = float(avg_fit)
                    fitness_data.append({
                        "generation": generation,
                        "best_fitness": best_fitness,
                        "avg_fitness": avg_fitness
                    })
                except ValueError:
                    pass
        i += 1

    # Build the JSON object
    algorithm_name = "differential_evolution" if "de" in txt_file_path else "cma-es"
    data["algorithm"] = algorithm_name

    # Extract info from header
    for key, value in header_info.items():
        if key not in ["Elapsed_Time", "Generations", "Steps"]:
            data[key.lower()] = value

    data["generations"] = header_info.get("Generations", len(fitness_data))
    data["steps"] = header_info.get("Steps", 500)
    data["elapsed_time"] = elapsed_time

    # Get best fitness from the last generation
    best_fitness_overall = max(entry["best_fitness"]
                               for entry in fitness_data) if fitness_data else 0
    data["best fitness"] = best_fitness_overall

    # Calculate total evaluations (population_size * generations)
    population_size = header_info.get("Population_Size", 50)
    generations = header_info.get("Generations", len(fitness_data))
    data["total_eval"] = population_size * generations

    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(txt_file_path)

    # Create output filenames
    json_path = os.path.join(output_dir, 'data.json')
    csv_path = os.path.join(output_dir, 'fitness_history.csv')

    # Create DataFrame for CSV
    df = pd.DataFrame([
        {
            'Iteration': entry['generation'],
            'Best Reward': entry['best_fitness'],
            'Mean Reward': entry['avg_fitness'],
        }
        for entry in fitness_data
    ])

    # Write to JSON file
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=False)

    # Write to CSV file
    df.to_csv(csv_path, index=False)

    print(f"Converted {txt_file_path} to:")
    print(f"  - JSON: {json_path}")
    print(f"  - CSV: {csv_path}")

    return data, df


# Example usage
if __name__ == "__main__":
    # To process all results.txt files

    for root, dirs, files in os.walk("final_results/3_2"):
        for file in files:
            if file == "results.txt":
                txt_file_path = os.path.join(root, file)
                extract_results_to_json_and_csv(txt_file_path)
