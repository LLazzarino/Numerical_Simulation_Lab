#!/usr/bin/env python
import subprocess
import os
import numpy as np

# Define the number of times you want to run the program
num_runs = 50

# Define the path to your C++ program
cpp_program = "./Monte_Carlo_ISING_1D.exe"

# Define the path to your input file template
input_template = "./input_template/input.dat"

# Define the directory where the modified input files will be stored
#output_dir = "./script_test/"

# Create the output directory if it doesn't exist
#os.makedirs(output_dir, exist_ok=True)

temps = np.linspace(0.5,2.0,num=num_runs)

for i in range(num_runs):
    # Generate a new input file
    new_input_file = "input.dat"
    with open(input_template, 'r') as template_file:
        template_contents = template_file.read()
    # Modify the input file (you can customize this part)
    # For example, you can randomly generate values and replace placeholders in the template
    modified_contents = template_contents.replace("0.5", f"{temps[i]}")
    with open(new_input_file, 'w') as new_input:
        new_input.write(modified_contents)

    # Execute the C++ program with the modified input file
    subprocess.run(cpp_program)
    print(f"{i} th run completed")


# Define the path to the output file
output_files = ["./output.ene(T).txt","./output.heat(T).txt","./output.mag(T).txt","./output.chi(T).txt"]
simulation_outputs = ["./output.ene.0","./output.heat.0","./output.mag.0","./output.chi.0"]

nblocks = 30

# Read and save every 1000th line from the output file
for i in range(4):
    with open(output_files[i], 'a') as output:
        with open(simulation_outputs[i], 'r') as f:
            lines = f.readlines()
            for line_num, line in enumerate(lines):
                if (line_num-1) % nblocks == 0:
                    output.write(line)

print("All runs completed.")