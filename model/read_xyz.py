import numpy as np


def read_xyz_file(filename):
    with open(filename, 'r') as file:
        #
        num_atoms = int(file.readline().strip())

        #
        atoms = []
        for i in range(num_atoms):
            line = file.readline().strip().split()
            symbol = line[0]
            x, y, z = map(float, line[1:])
            atoms.append({'symbol': symbol, 'x': x, 'y': y, 'z': z})

    return atoms


# 使用示例

# atoms = read_xyz_file(filename)
# for atom in atoms:
#     print(f'Symbol: {atom["symbol"]}, X: {atom["x"]}, Y: {atom["y"]}, Z: {atom["z"]}')


