import numpy as np
import pandas as pd
from Compute import computeCoordinateMissingIC
from Compute import computeCoordinateMissingICSolve

def computeEuclideanDistancesMatrix(x):
    x_square = np.expand_dims(np.einsum('ij,ij->i', x, x), axis=1)
    distances = np.dot(x, x.T)
    # use inplace operation to accelerate
    distances *= -2
    distances += x_square
    distances += x_square
    # result maybe less than 0 due to floating point rounding errors.
    np.maximum(distances, 0, distances)
    distances.flat[::distances.shape[0] + 1] = 0.0
    return distances


class LammpsAtom():

    def __init__(self, dtype, mole, coord, bonded=None):

        # atom_style angle
        self.dtype = int(dtype)  # Must be int
        self.mole = int(mole) # Must be int
        self.coord = np.array(coord).astype('float')
        self.bonded = bonded

    def regular_output(self):

        # molecule-id atom-type x y z
        return [self.mole, self.dtype] + self.coord.tolist()


NM2ANG = 10
KJ2KCAL = 1/4.184
RAD2DEG = 180/np.pi

class LammpsData():

    def __init__(self, ):

        self.content = []

    def add_atom(self, atom):

        self.content.append(atom)

    def build_atoms(self, mass_dict, pair_dict):

        columns = ["molecule-id", "atom-type",  "x", "y", "z"]
        self.atoms_df = pd.DataFrame(
            [*map(lambda x: x.regular_output(), self.content)], columns=columns)
        self.atoms_df.index = self.atoms_df.index+1
        self.atoms_df.index.name = "atom-id"
        
        self.natoms_types = len(np.unique(self.atoms_df.iloc[:,1]))
        
        # mass unit is g/mol
        self.mass_df = pd.DataFrame([(dtype + 1, mass_dict[dtype + 1])
                                     for dtype in range(self.natoms_types)],
                                    columns=['atom-type', 'mass'])
        self.mass_df.index = self.mass_df['atom-type']
        self.mass_df.index.name = "atom-type"
        self.mass_df.drop(columns=['atom-type'], inplace=True)
        
        # 96/cut have a different definition of eps
        # distance unit is A, energy unit is kcal/mol
        self.pairs_coeff_df = pd.DataFrame([(dtype + 1, 
                                             pair_dict[dtype + 1][0]/16*27 * KJ2KCAL, 
                                             pair_dict[dtype + 1][1] * NM2ANG
                                             ) for dtype in range(self.natoms_types)],
                                    columns=['atom-type', 'eps', 'sigma'])
        self.pairs_coeff_df.index = self.pairs_coeff_df['atom-type']
        self.pairs_coeff_df.index.name = "atom-type"
        self.pairs_coeff_df.drop(columns=['atom-type'], inplace=True)
        
    def build_bonds(self, bond_dict):

        self.bond_types = set()
        for atom1 in self.content:
            if atom1.bonded is None: continue
            for atom2 in atom1.bonded:
                self.bond_types.add((atom1.dtype, atom2.dtype
                                     ) if atom1.dtype < atom2.dtype else (
                                     atom2.dtype, atom1.dtype))
                                         
        self.bond_types = list(self.bond_types)[::-1]

        # Step 2: Iterate through each LammpsAtom and create bonds DataFrame
        self.bond_data = []
        for atom1 in self.content:
            if atom1.bonded is None: continue
            for atom2 in atom1.bonded:
                # Step 3: Add the bond data to the bond_data list
                bond_type = (atom1.dtype, atom2.dtype
                             ) if atom1.dtype < atom2.dtype else (
                             atom2.dtype, atom1.dtype)
                self.bond_data.append([self.bond_types.index(bond_type) + 1,
                                       self.content.index(atom1) + 1,
                                       self.content.index(atom2) + 1,
                                       ])

        # Create a DataFrame with bond data
        columns = ["bond-type", "atom-1", "atom-2"]
        self.bonds_df = pd.DataFrame(self.bond_data, columns=columns)
        self.bonds_df = self.bonds_df.sort_values(by='bond-type').reset_index(drop=True)
        self.bonds_df.index += 1
        self.bonds_df.index.names = ['bond-id']
        
        # bond style harmonic do not contains 1/2
        # distance unit is A, energy unit is kcal/mol
        self.bonds_coeff_df = pd.DataFrame([(bond_dict[bond_type][0]/2 * KJ2KCAL/(NM2ANG)**2, 
                                             bond_dict[bond_type][1]*NM2ANG
                                             )  for bond_type in self.bond_types], 
                                           columns=['Kb', 'l0'])
        self.bonds_coeff_df.index += 1
        self.bonds_coeff_df.index.names = ['bond-type'] 

    def build_angles(self, angle_dict):

        self.angle_types = set()
        for atom1 in self.content:
            if atom1.bonded is None: continue
            for atom2 in atom1.bonded:
                if atom2.bonded is None: continue
                for atom3 in atom2.bonded:
                    if atom3 == atom1: continue
                    self.angle_types.add((atom1.dtype, atom2.dtype, atom3.dtype
                                          ) if atom1.dtype < atom3.dtype else (
                                          atom3.dtype, atom2.dtype, atom1.dtype))
                                              
        self.angle_types = list(self.angle_types)[::-1]

        # Initialize a list to store angle data
        self.angle_data = []
        for atom1 in self.content:
            if atom1.bonded is None: continue
            for atom2 in atom1.bonded:
                if atom2.bonded is None: continue
                for atom3 in atom2.bonded:
                    if atom3 == atom1: continue
                    angle_type = (atom1.dtype, atom2.dtype, atom3.dtype
                                  ) if atom1.dtype < atom3.dtype else (
                                  atom3.dtype, atom2.dtype, atom1.dtype)
                    self.angle_data.append([self.angle_types.index(angle_type) + 1,
                                            self.content.index(atom1) + 1,
                                            self.content.index(atom2) + 1,
                                            self.content.index(atom3) + 1])

        # Create a DataFrame with angle data
        columns = ["angle-type", "atom-1", "atom-2", "atom-3"]
        self.angles_df = pd.DataFrame(self.angle_data, columns=columns)
        self.angles_df = self.angles_df.sort_values(by='angle-type').reset_index(drop=True)
        self.angles_df.index += 1
        self.angles_df.index.names = ['angle-id']

        # angle style harmonic do not contains 1/2, 
        # angle unit is in degree instead of rad, energy unit in kcal
        self.angles_coeff_df = pd.DataFrame([(angle_dict[angle_type][0]/2*KJ2KCAL,
                                              angle_dict[angle_type][1]
                                              ) for angle_type in self.angle_types],
                                            columns=['Ka', 'theta0'])
        self.angles_coeff_df.index = self.angles_coeff_df.index + 1
        self.angles_coeff_df.index.names = ['angle-type']


    def output_file(self, filename='TestOut.lammps', limit=None):

        if limit==None:
            x_max, x_min = self.atoms_df['x'].max(), self.atoms_df['x'].min()
            y_max, y_min = self.atoms_df['y'].max(), self.atoms_df['y'].min()
            z_max, z_min = self.atoms_df['z'].max(), self.atoms_df['z'].min()
        else:
            x_min, x_max, y_min, y_max, z_min, z_max = limit

        with open(filename, 'w') as f:
            
            # Write the title line
            f.write("LAMMPS Description\n")
            f.write("\n")
           
            # Write the number of atoms, bonds, angles, dihedrals, and impropers
            f.write(f"{len(self.content)} atoms\n")
            f.write(f"{len(self.bonds_df)} bonds\n")
            f.write(f"{len(self.angles_df)} angles\n")
            f.write("\n")

            # Write the number of atom, bond, angle, dihedral, and improper types
            f.write(f"{self.natoms_types} atom types\n")
            f.write(f"{len(self.bonds_coeff_df)} bond types\n")
            f.write(f"{len(self.angles_coeff_df)} angle types\n")
            f.write("\n")

            # Write the box size or min/max extent of atoms
            f.write(f"{x_min} {x_max} xlo xhi\n")
            f.write(f"{y_min} {y_max} ylo yhi\n")
            f.write(f"{z_min} {z_max} zlo zhi\n")
            f.write("\n")
            
            # Write Masses
            f.write("Masses\n\n")
            mass_df_string = self.mass_df.to_string(index=True, header=False, 
                                                    index_names=False, justify='left')
            f.write(mass_df_string + "\n\n")

            # Write Pair Coeffs
            f.write("Pair Coeffs # lj96/cut \n\n")
            pair_coeff_df_string = self.pairs_coeff_df.to_string(index=True, 
                                                                header=False, 
                                                                index_names=False, 
                                                                justify='left')
            f.write(pair_coeff_df_string + "\n\n")

            # Write Bond Coeffs
            f.write("Bond Coeffs # harmonic \n\n")
            bonds_coeff_df_string = self.bonds_coeff_df.to_string(index=True, 
                                                                  header=False, 
                                                                  index_names=False, 
                                                                  justify='left')
            f.write(bonds_coeff_df_string + "\n\n")

            # Write Angle Coeffs
            f.write("Angle Coeffs # harmonic \n\n")
            angles_coeff_df_string = self.angles_coeff_df.to_string(index=True, 
                                                                    header=False, 
                                                                    index_names=False, 
                                                                    justify='left')
            f.write(angles_coeff_df_string + "\n\n")
            
            # Write Atoms
            f.write("Atoms # angle\n\n")
            atoms_df_string = self.atoms_df.to_string(index=True, 
                                                      header=False, 
                                                      index_names=False, 
                                                      justify='left')
            f.write(atoms_df_string + "\n\n")

            # Write Bonds
            f.write("Bonds\n\n")
            bonds_df_string = self.bonds_df.to_string(index=True, 
                                                      header=False, 
                                                      index_names=False, 
                                                      justify='left')
            f.write(bonds_df_string + "\n\n")

            # Write Angles
            f.write("Angles\n\n")
            angles_df_string = self.angles_df.to_string(index=True, 
                                                        header=False, 
                                                        index_names=False, 
                                                        justify='left')
            f.write(angles_df_string + "\n\n")

            self.to_xyz(filename.split('.')[0])

    def to_xyz(self, name):
        df = self.atoms_df
        xyz_lines = [str(len(df)), '']  # Include the title line
        for _, row in df.iterrows():
            atom_type = int(row['atom-type'])
            x, y, z = row['x'], row['y'], row['z']
            xyz_line = f"{atom_type} {x:.6f} {y:.6f} {z:.6f}"
            xyz_lines.append(xyz_line)
        xyz = "\n".join(xyz_lines)
        output_file_path = name + '.xyz'
        with open(output_file_path, 'w') as output_file:
            output_file.write(xyz)

            

'''
if __name__ == '__main__':
    
    
    L = 130
    lammps_data = LammpsData()
    
    n = np.loadtxt('sphere_loc.txt')
    # 1M, 2C2, 3C3, 4P3, 5O, 6C6, 7CN, 8Oxy
        
    mass_dict = {1: 100.00,      # fixed
                 2: 28.054,
                 3: 42.081, 
                 4: 39.057,
                 5: 15.999,
                 6: 78.114,
                 7: 26.018,
                 8: 15.999,
                 9: 78.114,
                 10: 100.00
                 }
    
    bond_dict = {(1,1):(0,        0.5000),
                 (1,2):(0,        0.4000),
                 (2,2):(20000,    0.2550),
                 (2,3):(9600,     0.2900),
                 (3,3):(5892,     0.3392),
                 (3,4):(20000,    0.2090),
                 (4,4):(20000,    0.1866),
                 (4,5):(20000,    0.1853),
                 (2,5):(20000,    0.1811),
                 (4,8):(20000,    0.1853),
                 (2,8):(20000,    0.1811),
                 (4,6):(20000,    0.3080),
                 (4,9):(20000,    0.3080),
                 (4,7):(20000,    0.2420),
                 }
    
    angle_dict = {(1,1,1): (0,      0),
                  (1,1,2): (0,      0),
                  (1,2,2): (0,    180), # fixed
                  (2,2,3): (17,   180),
                  (2,3,2): (15.4, 180),
                  (3,3,3): (28,3, 146),
                  (2,3,3): (30,   152),
                  (2,2,2): (30,   146),
                  (3,3,4): (35,   143),
                  (3,4,4): (36.8, 157.5),
                  (4,4,5): (380,  180),  # fixed 
                  (4,4,8): (380,  100),  # fixed   
                  (5,4,6): (380,  180),  # fixed
                  (4,6,4): (380,  180),  # fixed
                  (4,9,4): (380,  180),  # fixed
                  (7,4,9): (360,  180),
                  (3,4,9): (95,   163),
                  (8,4,8): (380,  160),
                  (5,4,8): (380,   80),
                  (2,3,4): (50,   143),
                  (2,5,4): (0,     2031),   # tabulated, C2-Op-P3
                  (2,2,5): (0,     220),   # tabulated, C2-C2-O
                  (2,8,4): (0,     2023),   # tabulated, C2-Om-P3
                  (2,2,8): (0,     220),   # tabulated, C2-C2-O
                  (5,2,5): (0,     2),   # tabulated, O-C2-O
                  (2,5,2): (0,     202),   # tabulated, C2-O-C2
                  (5,2,8): (0,     2),   # tabulated, O-C2-O              
                  }
                  
    pair_dict = {1: (0,      0.000),     # M, special WCA
                 2: (1.0667, 0.430),     # C2
                 3: (1.8963, 0.470),     # C3
                 4: (1.4815, 0.460),     # P3
                 5: (0.8600, 0.372),     # O
                 6: (3.5556, 0.545),     # C6
                 7: (1.8963, 0.360),     # CN
                 8: (0.8600, 0.372),     # O
                 9: (3.5556, 0.545),     # C6
                 10: (0.000, 0.000)
                 }                 
             
    
    
    Mlist = []
    d = computeEuclideanDistancesMatrix(n)
    d = np.sqrt(d)
    d[d==0] = 100
    index = np.argwhere(d < 0.6)
    for i, loc in enumerate(n):
        Mlist.append(LammpsAtom(1, 1, loc*10, []))
    for i, j in index:
        Mlist[i].bonded += [Mlist[j]]
    lammps_data.content += Mlist   
    n2 = np.loadtxt('167_loc2.txt')
    ind = [np.argwhere(n==l)[0][0] for l in n2]
    
    
    for i, vec in enumerate(n):
        M = Mlist[i]
        Mlist.append(M)
        if i not in ind: continue
        vec = vec/np.linalg.norm(vec)
        C21 = LammpsAtom(2,   1, vec*29.5, [M]) #25.5 radius, 4 A bond length
        C22 = LammpsAtom(2,   1, vec*32.0, [C21]) #2.5 A bond length
        C23 = LammpsAtom(2, i+2, vec*34.5, [C22]) #2.5 A bond length
        C31 = LammpsAtom(3, i+2, vec*37.5, [C23]) #3.0 A bond length
        C32 = LammpsAtom(3, i+2, vec*41.0, [C31]) #3.5 A bond length
        P31 = LammpsAtom(4, i+2, vec*43.0, [C32]) #2.0 A bond length
        P32 = LammpsAtom(4, i+2, vec*(43.0+1.866), [P31]) # must be accurate A bond length
        O01 = LammpsAtom(5, i+2, vec*(43.0+1.853+1.866), [P32]) # must be accurate A bond length
        A,B,P = vec*43.0, vec*(43.0+1.866),vec*(43.0+1.853+1.866)
        C = computeCoordinateMissingICSolve(B,A,B, 1.853,180-100,0)
        D = computeCoordinateMissingIC(C, A, B, 1.853, 180-100, 180)
        O02 = LammpsAtom(8, i+2, C, [P32])
        O03 = LammpsAtom(8, i+2, D, [P32])
        P32.bonded = [P31, O01, O02, O03] # A fixer
        v1,v2,v3 = vec, (C-B)/np.linalg.norm(C-B), (D-B)/np.linalg.norm(D-B)
        lammps_data.content += [C21, C22, C23, C31, C32, P31, P32, O01, O02, O03]
        
        
        # write branch 1
        C211 = LammpsAtom(2, len(n)+2+3*i+0, P+v1*2.0, [O01])
        O011 = LammpsAtom(5, len(n)+2+3*i+0, P+v1*4.0, [C211])
        C212 = LammpsAtom(2, len(n)+2+3*i+0, P+v1*6.0, [O011])
        O012 = LammpsAtom(5, len(n)+2+3*i+0, P+v1*8.0, [C212])
        C213 = LammpsAtom(2, len(n)+2+3*i+0, P+v1*10.0, [O012])
        O013 = LammpsAtom(5, len(n)+2+3*i+0, P+v1*12.0, [C213])
        P311 = LammpsAtom(4, len(n)+2+3*i+0, P+v1*16.0, [O013])
        C611 = LammpsAtom(6, len(n)+2+3*i+0, P+v1*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, len(n)+2+3*i+0, P+v1*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, O011, C212, O012, C213, O013, P311, C611, P312]

        C211 = LammpsAtom(2, len(n)+2+3*i+1, C+v2*2.0, [O02])
        O011 = LammpsAtom(5, len(n)+2+3*i+1, C+v2*4.0, [C211])
        C212 = LammpsAtom(2, len(n)+2+3*i+1, C+v2*6.0, [O011])
        O012 = LammpsAtom(5, len(n)+2+3*i+1, C+v2*8.0, [C212])
        C213 = LammpsAtom(2, len(n)+2+3*i+1, C+v2*10.0, [O012])
        O013 = LammpsAtom(5, len(n)+2+3*i+1, C+v2*12.0, [C213])
        P311 = LammpsAtom(4, len(n)+2+3*i+1, C+v2*16.0, [O013])
        C611 = LammpsAtom(6, len(n)+2+3*i+1, C+v2*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, len(n)+2+3*i+1, C+v2*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, O011, C212, O012, C213, O013, P311, C611, P312]

        C211 = LammpsAtom(2, len(n)+2+3*i+2, D+v3*2.0, [O03])
        O011 = LammpsAtom(5, len(n)+2+3*i+2, D+v3*4.0, [C211])
        C212 = LammpsAtom(2, len(n)+2+3*i+2, D+v3*6.0, [O011])
        O012 = LammpsAtom(5, len(n)+2+3*i+2, D+v3*8.0, [C212])
        C213 = LammpsAtom(2, len(n)+2+3*i+2, D+v3*10.0, [O012])
        O013 = LammpsAtom(5, len(n)+2+3*i+2, D+v3*12.0, [C213])
        P311 = LammpsAtom(4, len(n)+2+3*i+2, D+v3*16.0, [O013])
        C611 = LammpsAtom(6, len(n)+2+3*i+2, D+v3*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, len(n)+2+3*i+2, D+v3*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, O011, C212, O012, C213, O013, P311, C611, P312]
    
    j = 2000
    # move to a second NC-L2  
   
    shifted = np.array([L/2, np.sqrt(3)*L/2, 0]) #30
    Mlist = []
    d = computeEuclideanDistancesMatrix(n)
    d = np.sqrt(d)
    d[d==0] = 100
    index = np.argwhere(d < 0.6)
    for i, loc in enumerate(n):
        Mlist.append(LammpsAtom(1, j, shifted+loc*10, []))
    for i, j in index:
        Mlist[i].bonded += [Mlist[j]]
    lammps_data.content += Mlist
   
    j = 2000
   
    for i, vec in enumerate(n):
        M = Mlist[i]
        Mlist.append(M)
        if i not in ind: continue
        vec = vec/np.linalg.norm(vec)
        C21 = LammpsAtom(2,   j, shifted+vec*29.5, [M]) #25.5 radius, 4 A bond length
        C22 = LammpsAtom(2,   j, shifted+vec*32.0, [C21]) #2.5 A bond length
        C23 = LammpsAtom(2, j+i+2, shifted+vec*34.5, [C22]) #2.5 A bond length
        C31 = LammpsAtom(3, j+i+2, shifted+vec*37.5, [C23]) #3.0 A bond length
        C32 = LammpsAtom(3, j+i+2, shifted+vec*41.0, [C31]) #3.5 A bond length
        P31 = LammpsAtom(4, j+i+2, shifted+vec*43.0, [C32]) #2.0 A bond length
        P32 = LammpsAtom(4, j+i+2, shifted+vec*(43.0+1.866), [P31]) # must be accurate A bond length
        O01 = LammpsAtom(5, j+i+2, shifted+vec*(43.0+1.853+1.866), [P32]) # must be accurate A bond length
        A,B,P = vec*43.0, vec*(43.0+1.866),vec*(43.0+1.853+1.866)
        C = computeCoordinateMissingICSolve(B,A,B, 1.853,180-100,0)
        D = computeCoordinateMissingIC(C, A, B, 1.853, 180-100, 180)
        O02 = LammpsAtom(8, j+i+2, shifted+C, [P32])
        O03 = LammpsAtom(8, j+i+2, shifted+D, [P32])
        P32.bonded = [P31, O01, O02, O03] # A fixer
        v1,v2,v3 = vec, (C-B)/np.linalg.norm(C-B), (D-B)/np.linalg.norm(D-B)
        lammps_data.content += [C21, C22, C23, C31, C32, P31, P32, O01, O02, O03]
       
        # write branch 1
        C211 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*2.0, [O01])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+0, shifted+P+v1*4.0, [C211])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*6.0, [O011])
        O012 = LammpsAtom(5, j+len(n)+2+3*i+0, shifted+P+v1*8.0, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*10.0, [O012])
        O013 = LammpsAtom(5, j+len(n)+2+3*i+0, shifted+P+v1*12.0, [C213])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+0, shifted+P+v1*16.0, [O013])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+0, shifted+P+v1*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+0, shifted+P+v1*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, O011, C212, O012, C213, O013, P311, C611, P312]
        
        # write branch 2
        C211 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*2.0, [O02])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+1, shifted+C+v2*4.0, [C211])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*6.0, [O011])
        O012 = LammpsAtom(5, j+len(n)+2+3*i+1, shifted+C+v2*8.0, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*10.0, [O012])
        O013 = LammpsAtom(5, j+len(n)+2+3*i+1, shifted+C+v2*12.0, [C213])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+1, shifted+C+v2*16.0, [O013])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+1, shifted+C+v2*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+1, shifted+C+v2*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, O011, C212, O012, C213, O013, P311, C611, P312]

        # write branch 3
        C211 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*2.0, [O03])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+2, shifted+D+v3*4.0, [C211])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*6.0, [O011])
        O012 = LammpsAtom(5, j+len(n)+2+3*i+2, shifted+D+v3*8.0, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*10.0, [O012])
        O013 = LammpsAtom(5, j+len(n)+2+3*i+2, shifted+D+v3*12.0, [C213])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+2, shifted+D+v3*16.0, [O013])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+2, shifted+D+v3*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+2, shifted+D+v3*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, O011, C212, O012, C213, O013, P311, C611, P312]


    j = 4000
    # move to a second NC-L2  
   
    #shifted = np.array([1/2*L,0,1/2*L]) #30
    shifted = np.array([0, np.sqrt(3)*(1)/3*L, -L*np.sqrt(6)/3])
    Mlist = []
    d = computeEuclideanDistancesMatrix(n)
    d = np.sqrt(d)
    d[d==0] = 100
    index = np.argwhere(d < 0.6)
    for i, loc in enumerate(n):
        Mlist.append(LammpsAtom(1, j, shifted+loc*10, []))
    for i, j in index:
        Mlist[i].bonded += [Mlist[j]]
    lammps_data.content += Mlist
   
    j = 4000
   
    for i, vec in enumerate(n):
        M = Mlist[i]
        Mlist.append(M)
        if i not in ind: continue
        vec = vec/np.linalg.norm(vec)
        C21 = LammpsAtom(2,   j, shifted+vec*29.5, [M]) #25.5 radius, 4 A bond length
        C22 = LammpsAtom(2,   j, shifted+vec*32.0, [C21]) #2.5 A bond length
        C23 = LammpsAtom(2, j+i+2, shifted+vec*34.5, [C22]) #2.5 A bond length
        C31 = LammpsAtom(3, j+i+2, shifted+vec*37.5, [C23]) #3.0 A bond length
        C32 = LammpsAtom(3, j+i+2, shifted+vec*41.0, [C31]) #3.5 A bond length
        P31 = LammpsAtom(4, j+i+2, shifted+vec*43.0, [C32]) #2.0 A bond length
        P32 = LammpsAtom(4, j+i+2, shifted+vec*(43.0+1.866), [P31]) # must be accurate A bond length
        O01 = LammpsAtom(5, j+i+2, shifted+vec*(43.0+1.853+1.866), [P32]) # must be accurate A bond length
        A,B,P = vec*43.0, vec*(43.0+1.866),vec*(43.0+1.853+1.866)
        C = computeCoordinateMissingICSolve(B,A,B, 1.853,180-100,0)
        D = computeCoordinateMissingIC(C, A, B, 1.853, 180-100, 180)
        O02 = LammpsAtom(8, j+i+2, shifted+C, [P32])
        O03 = LammpsAtom(8, j+i+2, shifted+D, [P32])
        P32.bonded = [P31, O01, O02, O03] # A fixer
        v1,v2,v3 = vec, (C-B)/np.linalg.norm(C-B), (D-B)/np.linalg.norm(D-B)
        lammps_data.content += [C21, C22, C23, C31, C32, P31, P32, O01, O02, O03]
       
        # write branch 1
        C211 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*2.0, [O01])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+0, shifted+P+v1*4.0, [C211])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*6.0, [O011])
        O012 = LammpsAtom(5, j+len(n)+2+3*i+0, shifted+P+v1*8.0, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*10.0, [O012])
        O013 = LammpsAtom(5, j+len(n)+2+3*i+0, shifted+P+v1*12.0, [C213])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+0, shifted+P+v1*16.0, [O013])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+0, shifted+P+v1*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+0, shifted+P+v1*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, O011, C212, O012, C213, O013, P311, C611, P312]
        
        # write branch 2
        C211 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*2.0, [O02])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+1, shifted+C+v2*4.0, [C211])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*6.0, [O011])
        O012 = LammpsAtom(5, j+len(n)+2+3*i+1, shifted+C+v2*8.0, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*10.0, [O012])
        O013 = LammpsAtom(5, j+len(n)+2+3*i+1, shifted+C+v2*12.0, [C213])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+1, shifted+C+v2*16.0, [O013])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+1, shifted+C+v2*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+1, shifted+C+v2*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, O011, C212, O012, C213, O013, P311, C611, P312]

        # write branch 3
        C211 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*2.0, [O03])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+2, shifted+D+v3*4.0, [C211])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*6.0, [O011])
        O012 = LammpsAtom(5, j+len(n)+2+3*i+2, shifted+D+v3*8.0, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*10.0, [O012])
        O013 = LammpsAtom(5, j+len(n)+2+3*i+2, shifted+D+v3*12.0, [C213])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+2, shifted+D+v3*16.0, [O013])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+2, shifted+D+v3*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+2, shifted+D+v3*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, O011, C212, O012, C213, O013, P311, C611, P312]


    j = 6000
    # move to a second NC-L2  
   
    #shifted = np.array([1/2*L,1/2*L,0]) #30
    shifted = np.array([1/2*L, -np.sqrt(3)*1/6*L, -L*np.sqrt(6)/3])
    Mlist = []
    d = computeEuclideanDistancesMatrix(n)
    d = np.sqrt(d)
    d[d==0] = 100
    index = np.argwhere(d < 0.6)
    for i, loc in enumerate(n):
        Mlist.append(LammpsAtom(1, j, shifted+loc*10, []))
    for i, j in index:
        Mlist[i].bonded += [Mlist[j]]
    lammps_data.content += Mlist
   
    j = 6000
   
    for i, vec in enumerate(n):
        M = Mlist[i]
        Mlist.append(M)
        if i not in ind: continue
        vec = vec/np.linalg.norm(vec)
        C21 = LammpsAtom(2,   j, shifted+vec*29.5, [M]) #25.5 radius, 4 A bond length
        C22 = LammpsAtom(2,   j, shifted+vec*32.0, [C21]) #2.5 A bond length
        C23 = LammpsAtom(2, j+i+2, shifted+vec*34.5, [C22]) #2.5 A bond length
        C31 = LammpsAtom(3, j+i+2, shifted+vec*37.5, [C23]) #3.0 A bond length
        C32 = LammpsAtom(3, j+i+2, shifted+vec*41.0, [C31]) #3.5 A bond length
        P31 = LammpsAtom(4, j+i+2, shifted+vec*43.0, [C32]) #2.0 A bond length
        P32 = LammpsAtom(4, j+i+2, shifted+vec*(43.0+1.866), [P31]) # must be accurate A bond length
        O01 = LammpsAtom(5, j+i+2, shifted+vec*(43.0+1.853+1.866), [P32]) # must be accurate A bond length
        A,B,P = vec*43.0, vec*(43.0+1.866),vec*(43.0+1.853+1.866)
        C = computeCoordinateMissingICSolve(B,A,B, 1.853,180-100,0)
        D = computeCoordinateMissingIC(C, A, B, 1.853, 180-100, 180)
        O02 = LammpsAtom(8, j+i+2, shifted+C, [P32])
        O03 = LammpsAtom(8, j+i+2, shifted+D, [P32])
        P32.bonded = [P31, O01, O02, O03] # A fixer
        v1,v2,v3 = vec, (C-B)/np.linalg.norm(C-B), (D-B)/np.linalg.norm(D-B)
        lammps_data.content += [C21, C22, C23, C31, C32, P31, P32, O01, O02, O03]
       
        # write branch 1
        C211 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*2.0, [O01])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+0, shifted+P+v1*4.0, [C211])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*6.0, [O011])
        O012 = LammpsAtom(5, j+len(n)+2+3*i+0, shifted+P+v1*8.0, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*10.0, [O012])
        O013 = LammpsAtom(5, j+len(n)+2+3*i+0, shifted+P+v1*12.0, [C213])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+0, shifted+P+v1*16.0, [O013])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+0, shifted+P+v1*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+0, shifted+P+v1*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, O011, C212, O012, C213, O013, P311, C611, P312]
        
        # write branch 2
        C211 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*2.0, [O02])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+1, shifted+C+v2*4.0, [C211])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*6.0, [O011])
        O012 = LammpsAtom(5, j+len(n)+2+3*i+1, shifted+C+v2*8.0, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*10.0, [O012])
        O013 = LammpsAtom(5, j+len(n)+2+3*i+1, shifted+C+v2*12.0, [C213])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+1, shifted+C+v2*16.0, [O013])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+1, shifted+C+v2*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+1, shifted+C+v2*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, O011, C212, O012, C213, O013, P311, C611, P312]

        # write branch 3
        C211 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*2.0, [O03])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+2, shifted+D+v3*4.0, [C211])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*6.0, [O011])
        O012 = LammpsAtom(5, j+len(n)+2+3*i+2, shifted+D+v3*8.0, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*10.0, [O012])
        O013 = LammpsAtom(5, j+len(n)+2+3*i+2, shifted+D+v3*12.0, [C213])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+2, shifted+D+v3*16.0, [O013])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+2, shifted+D+v3*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+2, shifted+D+v3*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, O011, C212, O012, C213, O013, P311, C611, P312]

   
    j = 8000
    # move to a second NC-L2  
   
    #shifted = np.array([1/2*L,1/2*L,0]) #30
    shifted = np.array([1/2*L, np.sqrt(3)*1/6*L, L*np.sqrt(6)/3])
    Mlist = []
    d = computeEuclideanDistancesMatrix(n)
    d = np.sqrt(d)
    d[d==0] = 100
    index = np.argwhere(d < 0.6)
    for i, loc in enumerate(n):
        Mlist.append(LammpsAtom(1, j, shifted+loc*10, []))
    for i, j in index:
        Mlist[i].bonded += [Mlist[j]]
    lammps_data.content += Mlist
    
    j = 8000
   
    for i, vec in enumerate(n):
        M = Mlist[i]
        Mlist.append(M)
        if i not in ind: continue
        vec = vec/np.linalg.norm(vec)
        C21 = LammpsAtom(2,   j, shifted+vec*29.5, [M]) #25.5 radius, 4 A bond length
        C22 = LammpsAtom(2,   j, shifted+vec*32.0, [C21]) #2.5 A bond length
        C23 = LammpsAtom(2, j+i+2, shifted+vec*34.5, [C22]) #2.5 A bond length
        C31 = LammpsAtom(3, j+i+2, shifted+vec*37.5, [C23]) #3.0 A bond length
        C32 = LammpsAtom(3, j+i+2, shifted+vec*41.0, [C31]) #3.5 A bond length
        P31 = LammpsAtom(4, j+i+2, shifted+vec*43.0, [C32]) #2.0 A bond length
        P32 = LammpsAtom(4, j+i+2, shifted+vec*(43.0+1.866), [P31]) # must be accurate A bond length
        O01 = LammpsAtom(5, j+i+2, shifted+vec*(43.0+1.853+1.866), [P32]) # must be accurate A bond length
        A,B,P = vec*43.0, vec*(43.0+1.866),vec*(43.0+1.853+1.866)
        C = computeCoordinateMissingICSolve(B,A,B, 1.853,180-100,0)
        D = computeCoordinateMissingIC(C, A, B, 1.853, 180-100, 180)
        O02 = LammpsAtom(8, j+i+2, shifted+C, [P32])
        O03 = LammpsAtom(8, j+i+2, shifted+D, [P32])
        P32.bonded = [P31, O01, O02, O03] # A fixer
        v1,v2,v3 = vec, (C-B)/np.linalg.norm(C-B), (D-B)/np.linalg.norm(D-B)
        lammps_data.content += [C21, C22, C23, C31, C32, P31, P32, O01, O02, O03]
       
        # write branch 1
        C211 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*2.0, [O01])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+0, shifted+P+v1*4.0, [C211])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*6.0, [O011])
        O012 = LammpsAtom(5, j+len(n)+2+3*i+0, shifted+P+v1*8.0, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*10.0, [O012])
        O013 = LammpsAtom(5, j+len(n)+2+3*i+0, shifted+P+v1*12.0, [C213])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+0, shifted+P+v1*16.0, [O013])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+0, shifted+P+v1*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+0, shifted+P+v1*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, O011, C212, O012, C213, O013, P311, C611, P312]
        
        # write branch 2
        C211 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*2.0, [O02])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+1, shifted+C+v2*4.0, [C211])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*6.0, [O011])
        O012 = LammpsAtom(5, j+len(n)+2+3*i+1, shifted+C+v2*8.0, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*10.0, [O012])
        O013 = LammpsAtom(5, j+len(n)+2+3*i+1, shifted+C+v2*12.0, [C213])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+1, shifted+C+v2*16.0, [O013])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+1, shifted+C+v2*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+1, shifted+C+v2*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, O011, C212, O012, C213, O013, P311, C611, P312]

        # write branch 3
        C211 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*2.0, [O03])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+2, shifted+D+v3*4.0, [C211])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*6.0, [O011])
        O012 = LammpsAtom(5, j+len(n)+2+3*i+2, shifted+D+v3*8.0, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*10.0, [O012])
        O013 = LammpsAtom(5, j+len(n)+2+3*i+2, shifted+D+v3*12.0, [C213])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+2, shifted+D+v3*16.0, [O013])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+2, shifted+D+v3*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+2, shifted+D+v3*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, O011, C212, O012, C213, O013, P311, C611, P312]

   
    j = 10000
    # move to a second NC-L2  
   
    #shifted = np.array([1/2*L,1/2*L,0]) #30
    shifted = np.array([0, np.sqrt(3)*2/3*L, L*np.sqrt(6)/3])
    Mlist = []
    d = computeEuclideanDistancesMatrix(n)
    d = np.sqrt(d)
    d[d==0] = 100
    index = np.argwhere(d < 0.6)
    for i, loc in enumerate(n):
        Mlist.append(LammpsAtom(1, j, shifted+loc*10, []))
    for i, j in index:
        Mlist[i].bonded += [Mlist[j]]
    lammps_data.content += Mlist
    
    j = 10000
   
   
    for i, vec in enumerate(n):
        M = Mlist[i]
        Mlist.append(M)
        if i not in ind: continue
        vec = vec/np.linalg.norm(vec)
        C21 = LammpsAtom(2,   j, shifted+vec*29.5, [M]) #25.5 radius, 4 A bond length
        C22 = LammpsAtom(2,   j, shifted+vec*32.0, [C21]) #2.5 A bond length
        C23 = LammpsAtom(2, j+i+2, shifted+vec*34.5, [C22]) #2.5 A bond length
        C31 = LammpsAtom(3, j+i+2, shifted+vec*37.5, [C23]) #3.0 A bond length
        C32 = LammpsAtom(3, j+i+2, shifted+vec*41.0, [C31]) #3.5 A bond length
        P31 = LammpsAtom(4, j+i+2, shifted+vec*43.0, [C32]) #2.0 A bond length
        P32 = LammpsAtom(4, j+i+2, shifted+vec*(43.0+1.866), [P31]) # must be accurate A bond length
        O01 = LammpsAtom(5, j+i+2, shifted+vec*(43.0+1.853+1.866), [P32]) # must be accurate A bond length
        A,B,P = vec*43.0, vec*(43.0+1.866),vec*(43.0+1.853+1.866)
        C = computeCoordinateMissingICSolve(B,A,B, 1.853,180-100,0)
        D = computeCoordinateMissingIC(C, A, B, 1.853, 180-100, 180)
        O02 = LammpsAtom(8, j+i+2, shifted+C, [P32])
        O03 = LammpsAtom(8, j+i+2, shifted+D, [P32])
        P32.bonded = [P31, O01, O02, O03] # A fixer
        v1,v2,v3 = vec, (C-B)/np.linalg.norm(C-B), (D-B)/np.linalg.norm(D-B)
        lammps_data.content += [C21, C22, C23, C31, C32, P31, P32, O01, O02, O03]
       
        # write branch 1
        C211 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*2.0, [O01])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+0, shifted+P+v1*4.0, [C211])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*6.0, [O011])
        O012 = LammpsAtom(5, j+len(n)+2+3*i+0, shifted+P+v1*8.0, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*10.0, [O012])
        O013 = LammpsAtom(5, j+len(n)+2+3*i+0, shifted+P+v1*12.0, [C213])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+0, shifted+P+v1*16.0, [O013])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+0, shifted+P+v1*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+0, shifted+P+v1*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, O011, C212, O012, C213, O013, P311, C611, P312]
        
        # write branch 2
        C211 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*2.0, [O02])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+1, shifted+C+v2*4.0, [C211])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*6.0, [O011])
        O012 = LammpsAtom(5, j+len(n)+2+3*i+1, shifted+C+v2*8.0, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*10.0, [O012])
        O013 = LammpsAtom(5, j+len(n)+2+3*i+1, shifted+C+v2*12.0, [C213])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+1, shifted+C+v2*16.0, [O013])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+1, shifted+C+v2*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+1, shifted+C+v2*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, O011, C212, O012, C213, O013, P311, C611, P312]

        # write branch 3
        C211 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*2.0, [O03])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+2, shifted+D+v3*4.0, [C211])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*6.0, [O011])
        O012 = LammpsAtom(5, j+len(n)+2+3*i+2, shifted+D+v3*8.0, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*10.0, [O012])
        O013 = LammpsAtom(5, j+len(n)+2+3*i+2, shifted+D+v3*12.0, [C213])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+2, shifted+D+v3*16.0, [O013])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+2, shifted+D+v3*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+2, shifted+D+v3*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, O011, C212, O012, C213, O013, P311, C611, P312]


    # This part is solvation
    j = 12000
    nx,ny,nz = 25, 44, 20
    spacing_x, spacing_y, spacing_z = 5, 5, 11
    for mol_id in range(nx*ny*nz):
        ix, iy, iz = mol_id % nx, (mol_id // nx) % ny, (mol_id // (nx * ny)) % nz       
        x, y, z = ix * spacing_x, iy * spacing_y, iz * spacing_z
        
        CN1 = LammpsAtom(7, mol_id+j, [x,y,z+200+ 0.0], None)
        P31 = LammpsAtom(4, mol_id+j, [x,y,z+200+ 2.5], [CN1])
        C61 = LammpsAtom(9, mol_id+j, [x,y,z+200+ 5.5], [P31])
        P32 = LammpsAtom(4, mol_id+j, [x,y,z+200+ 8.5], [C61])
        C31 = LammpsAtom(3, mol_id+j, [x,y+2, z+200+9], [P32])
        C21 = LammpsAtom(2, mol_id+j, [x,y+4, z+200+9], [C31])
        
        lammps_data.content += [CN1, P31, C61, P32, C31, C21]


    lammps_data.build_atoms(mass_dict, pair_dict)
    lammps_data.build_bonds(bond_dict)
    lammps_data.build_angles(angle_dict)
    lammps_data.output_file('L3-111Surface-Reduced.lammpsdata')
'''


if __name__ == '__main__':
    
    
    L = 130
    lammps_data = LammpsData()
    
    n = np.loadtxt('sphere_loc.txt')
    # 1M, 2C2, 3C3, 4P3, 5O, 6C6, 7CN, 8Oxy
        
    mass_dict = {1: 100.00,      # fixed
                 2: 28.054,
                 3: 42.081, 
                 4: 39.057,
                 5: 15.999,
                 6: 78.114,
                 7: 26.018,
                 8: 15.999,
                 9: 78.114,
                 10: 100.00
                 }
    
    bond_dict = {(1,1):(0,        0.5000),
                 (1,2):(0,        0.4000),
                 (2,2):(20000,    0.2550),
                 (2,3):(9600,     0.2900),
                 (3,3):(5892,     0.3392),
                 (3,4):(20000,    0.2090),
                 (4,4):(20000,    0.1866),
                 (4,5):(20000,    0.1853),
                 (2,5):(20000,    0.1811),
                 (4,8):(20000,    0.1853),
                 (2,8):(20000,    0.1811),
                 (4,6):(20000,    0.3080),
                 (4,9):(20000,    0.3080),
                 (4,7):(20000,    0.2420),
                 }
    
    angle_dict = {(1,1,1): (0,      0),
                  (1,1,2): (0,      0),
                  (1,2,2): (0,    180), # fixed
                  (2,2,3): (17,   180),
                  (2,3,2): (15.4, 180),
                  (3,3,3): (28,3, 146),
                  (2,3,3): (30,   152),
                  (2,2,2): (30,   146),
                  (3,3,4): (35,   143),
                  (3,4,4): (36.8, 157.5),
                  (4,4,5): (380,  180),  # fixed 
                  (4,4,8): (380,  100),  # fixed   
                  (5,4,6): (380,  180),  # fixed
                  (4,6,4): (380,  180),  # fixed
                  (4,9,4): (380,  180),  # fixed
                  (7,4,9): (360,  180),
                  (3,4,9): (95,   163),
                  (8,4,8): (380,  160),
                  (5,4,8): (380,   80),
                  (2,3,4): (50,   143),
                  (2,5,4): (0,     2031),   # tabulated, C2-Op-P3
                  (2,2,5): (0,     220),   # tabulated, C2-C2-O
                  (2,8,4): (0,     2023),   # tabulated, C2-Om-P3
                  (2,2,8): (0,     220),   # tabulated, C2-C2-O
                  (5,2,5): (0,     2),   # tabulated, O-C2-O
                  (2,5,2): (0,     202),   # tabulated, C2-O-C2
                  (5,2,8): (0,     2),   # tabulated, O-C2-O              
                  }
                  
    pair_dict = {1: (0,      0.000),     # M, special WCA
                 2: (1.0667, 0.430),     # C2
                 3: (1.8963, 0.470),     # C3
                 4: (1.4815, 0.460),     # P3
                 5: (0.8600, 0.372),     # O
                 6: (3.5556, 0.545),     # C6
                 7: (1.8963, 0.360),     # CN
                 8: (0.8600, 0.372),     # O
                 9: (3.5556, 0.545),     # C6
                 10: (0.000, 0.000)
                 }                 
             
    
    
    Mlist = []
    d = computeEuclideanDistancesMatrix(n)
    d = np.sqrt(d)
    d[d==0] = 100
    index = np.argwhere(d < 0.6)
    for i, loc in enumerate(n):
        Mlist.append(LammpsAtom(1, 1, loc*10, []))
    for i, j in index:
        Mlist[i].bonded += [Mlist[j]]
    lammps_data.content += Mlist   
    shifted = np.array([0,0,0])
    n2 = np.loadtxt('167_loc2.txt')
    ind = [np.argwhere(n==l)[0][0] for l in n2]
    
    for i, vec in enumerate(n):
        M = Mlist[i]
        Mlist.append(M)
        if i not in ind: continue
        vec = vec/np.linalg.norm(vec)
        C21 = LammpsAtom(2,   1, vec*29.5, [M]) #25.5 radius, 4 A bond length
        C22 = LammpsAtom(2,   1, vec*32.0, [C21]) #2.5 A bond length
        C23 = LammpsAtom(2, i+2, vec*34.5, [C22]) #2.5 A bond length
        C31 = LammpsAtom(3, i+2, vec*37.5, [C23]) #3.0 A bond length
        C32 = LammpsAtom(3, i+2, vec*41.0, [C31]) #3.5 A bond length
        P31 = LammpsAtom(4, i+2, vec*43.0, [C32]) #2.0 A bond length
        P32 = LammpsAtom(4, i+2, vec*(43.0+1.866), [P31]) # must be accurate A bond length
        O01 = LammpsAtom(5, i+2, vec*(43.0+1.853+1.866), [P32]) # must be accurate A bond length
        A,B,P = vec*43.0, vec*(43.0+1.866),vec*(43.0+1.853+1.866)
        C = computeCoordinateMissingICSolve(B,A,B, 1.853,180-100,0)
        D = computeCoordinateMissingIC(C, A, B, 1.853, 180-100, 180)
        O02 = LammpsAtom(8, i+2, C, [P32])
        O03 = LammpsAtom(8, i+2, D, [P32])
        P32.bonded = [P31, O01, O02, O03] # A fixer
        v1,v2,v3 = vec, (C-B)/np.linalg.norm(C-B), (D-B)/np.linalg.norm(D-B)
        lammps_data.content += [C21, C22, C23, C31, C32, P31, P32, O01, O02, O03]
        
        
        # write branch 1
        C211 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*2.0, [O01])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*4.5, [C211])
        C311 = LammpsAtom(3, j+len(n)+2+3*i+0, shifted+P+v1*7.5, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*10.0, [C311])
        C214 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*12.5, [C213])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+0, shifted+P+v1*14.0, [C214])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+0, shifted+P+v1*16.0, [O011])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+0, shifted+P+v1*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+0, shifted+P+v1*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, C212, C311, C213, C214, O011, P311, C611, P312]

        # write branch 2
        C211 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*2.0, [O02])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*4.5, [C211])
        C311 = LammpsAtom(3, j+len(n)+2+3*i+1, shifted+C+v2*7.5, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*10.0, [C311])
        C214 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*12.5, [C213])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+1, shifted+C+v2*14.0, [C214])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+1, shifted+C+v2*16.0, [O011])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+1, shifted+C+v2*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+1, shifted+C+v2*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, C212, C311, C213, C214, O011, P311, C611, P312]
        
        # write branch 3
        C211 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*2.0, [O03])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*4.5, [C211])
        C311 = LammpsAtom(3, j+len(n)+2+3*i+2, shifted+D+v3*7.5, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*10.0, [C311])
        C214 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*12.5, [C213])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+2, shifted+D+v3*14.0, [C214])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+2, shifted+D+v3*16.0, [O011])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+2, shifted+D+v3*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+2, shifted+D+v3*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, C212, C311, C213, C214, O011, P311, C611, P312]
    
    j = 2000
    # move to a second NC-L2  
   
    shifted = np.array([L/2, np.sqrt(3)*L/2, 0]) #30
    Mlist = []
    d = computeEuclideanDistancesMatrix(n)
    d = np.sqrt(d)
    d[d==0] = 100
    index = np.argwhere(d < 0.6)
    for i, loc in enumerate(n):
        Mlist.append(LammpsAtom(1, j, shifted+loc*10, []))
    for i, j in index:
        Mlist[i].bonded += [Mlist[j]]
    lammps_data.content += Mlist
   
    j = 2000
   
    for i, vec in enumerate(n):
        M = Mlist[i]
        Mlist.append(M)
        if i not in ind: continue
        vec = vec/np.linalg.norm(vec)
        C21 = LammpsAtom(2,   j, shifted+vec*29.5, [M]) #25.5 radius, 4 A bond length
        C22 = LammpsAtom(2,   j, shifted+vec*32.0, [C21]) #2.5 A bond length
        C23 = LammpsAtom(2, j+i+2, shifted+vec*34.5, [C22]) #2.5 A bond length
        C31 = LammpsAtom(3, j+i+2, shifted+vec*37.5, [C23]) #3.0 A bond length
        C32 = LammpsAtom(3, j+i+2, shifted+vec*41.0, [C31]) #3.5 A bond length
        P31 = LammpsAtom(4, j+i+2, shifted+vec*43.0, [C32]) #2.0 A bond length
        P32 = LammpsAtom(4, j+i+2, shifted+vec*(43.0+1.866), [P31]) # must be accurate A bond length
        O01 = LammpsAtom(5, j+i+2, shifted+vec*(43.0+1.853+1.866), [P32]) # must be accurate A bond length
        A,B,P = vec*43.0, vec*(43.0+1.866),vec*(43.0+1.853+1.866)
        C = computeCoordinateMissingICSolve(B,A,B, 1.853,180-100,0)
        D = computeCoordinateMissingIC(C, A, B, 1.853, 180-100, 180)
        O02 = LammpsAtom(8, j+i+2, shifted+C, [P32])
        O03 = LammpsAtom(8, j+i+2, shifted+D, [P32])
        P32.bonded = [P31, O01, O02, O03] # A fixer
        v1,v2,v3 = vec, (C-B)/np.linalg.norm(C-B), (D-B)/np.linalg.norm(D-B)
        lammps_data.content += [C21, C22, C23, C31, C32, P31, P32, O01, O02, O03]
       
        # write branch 1
        C211 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*2.0, [O01])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*4.5, [C211])
        C311 = LammpsAtom(3, j+len(n)+2+3*i+0, shifted+P+v1*7.5, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*10.0, [C311])
        C214 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*12.5, [C213])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+0, shifted+P+v1*14.0, [C214])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+0, shifted+P+v1*16.0, [O011])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+0, shifted+P+v1*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+0, shifted+P+v1*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, C212, C311, C213, C214, O011, P311, C611, P312]

        # write branch 2
        C211 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*2.0, [O02])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*4.5, [C211])
        C311 = LammpsAtom(3, j+len(n)+2+3*i+1, shifted+C+v2*7.5, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*10.0, [C311])
        C214 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*12.5, [C213])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+1, shifted+C+v2*14.0, [C214])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+1, shifted+C+v2*16.0, [O011])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+1, shifted+C+v2*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+1, shifted+C+v2*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, C212, C311, C213, C214, O011, P311, C611, P312]
        
        # write branch 3
        C211 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*2.0, [O03])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*4.5, [C211])
        C311 = LammpsAtom(3, j+len(n)+2+3*i+2, shifted+D+v3*7.5, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*10.0, [C311])
        C214 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*12.5, [C213])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+2, shifted+D+v3*14.0, [C214])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+2, shifted+D+v3*16.0, [O011])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+2, shifted+D+v3*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+2, shifted+D+v3*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, C212, C311, C213, C214, O011, P311, C611, P312]


    j = 4000
    # move to a second NC-L2  
   
    #shifted = np.array([1/2*L,0,1/2*L]) #30
    shifted = np.array([0, np.sqrt(3)*(1)/3*L, -L*np.sqrt(6)/3])
    Mlist = []
    d = computeEuclideanDistancesMatrix(n)
    d = np.sqrt(d)
    d[d==0] = 100
    index = np.argwhere(d < 0.6)
    for i, loc in enumerate(n):
        Mlist.append(LammpsAtom(1, j, shifted+loc*10, []))
    for i, j in index:
        Mlist[i].bonded += [Mlist[j]]
    lammps_data.content += Mlist
   
    j = 4000
   
    for i, vec in enumerate(n):
        M = Mlist[i]
        Mlist.append(M)
        if i not in ind: continue
        vec = vec/np.linalg.norm(vec)
        C21 = LammpsAtom(2,   j, shifted+vec*29.5, [M]) #25.5 radius, 4 A bond length
        C22 = LammpsAtom(2,   j, shifted+vec*32.0, [C21]) #2.5 A bond length
        C23 = LammpsAtom(2, j+i+2, shifted+vec*34.5, [C22]) #2.5 A bond length
        C31 = LammpsAtom(3, j+i+2, shifted+vec*37.5, [C23]) #3.0 A bond length
        C32 = LammpsAtom(3, j+i+2, shifted+vec*41.0, [C31]) #3.5 A bond length
        P31 = LammpsAtom(4, j+i+2, shifted+vec*43.0, [C32]) #2.0 A bond length
        P32 = LammpsAtom(4, j+i+2, shifted+vec*(43.0+1.866), [P31]) # must be accurate A bond length
        O01 = LammpsAtom(5, j+i+2, shifted+vec*(43.0+1.853+1.866), [P32]) # must be accurate A bond length
        A,B,P = vec*43.0, vec*(43.0+1.866),vec*(43.0+1.853+1.866)
        C = computeCoordinateMissingICSolve(B,A,B, 1.853,180-100,0)
        D = computeCoordinateMissingIC(C, A, B, 1.853, 180-100, 180)
        O02 = LammpsAtom(8, j+i+2, shifted+C, [P32])
        O03 = LammpsAtom(8, j+i+2, shifted+D, [P32])
        P32.bonded = [P31, O01, O02, O03] # A fixer
        v1,v2,v3 = vec, (C-B)/np.linalg.norm(C-B), (D-B)/np.linalg.norm(D-B)
        lammps_data.content += [C21, C22, C23, C31, C32, P31, P32, O01, O02, O03]
       
        # write branch 1
        C211 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*2.0, [O01])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*4.5, [C211])
        C311 = LammpsAtom(3, j+len(n)+2+3*i+0, shifted+P+v1*7.5, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*10.0, [C311])
        C214 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*12.5, [C213])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+0, shifted+P+v1*14.0, [C214])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+0, shifted+P+v1*16.0, [O011])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+0, shifted+P+v1*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+0, shifted+P+v1*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, C212, C311, C213, C214, O011, P311, C611, P312]

        # write branch 2
        C211 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*2.0, [O02])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*4.5, [C211])
        C311 = LammpsAtom(3, j+len(n)+2+3*i+1, shifted+C+v2*7.5, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*10.0, [C311])
        C214 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*12.5, [C213])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+1, shifted+C+v2*14.0, [C214])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+1, shifted+C+v2*16.0, [O011])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+1, shifted+C+v2*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+1, shifted+C+v2*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, C212, C311, C213, C214, O011, P311, C611, P312]
        
        # write branch 3
        C211 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*2.0, [O03])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*4.5, [C211])
        C311 = LammpsAtom(3, j+len(n)+2+3*i+2, shifted+D+v3*7.5, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*10.0, [C311])
        C214 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*12.5, [C213])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+2, shifted+D+v3*14.0, [C214])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+2, shifted+D+v3*16.0, [O011])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+2, shifted+D+v3*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+2, shifted+D+v3*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, C212, C311, C213, C214, O011, P311, C611, P312]

    j = 6000
    # move to a second NC-L2  
   
    #shifted = np.array([1/2*L,1/2*L,0]) #30
    shifted = np.array([1/2*L, -np.sqrt(3)*1/6*L, -L*np.sqrt(6)/3])
    Mlist = []
    d = computeEuclideanDistancesMatrix(n)
    d = np.sqrt(d)
    d[d==0] = 100
    index = np.argwhere(d < 0.6)
    for i, loc in enumerate(n):
        Mlist.append(LammpsAtom(1, j, shifted+loc*10, []))
    for i, j in index:
        Mlist[i].bonded += [Mlist[j]]
    lammps_data.content += Mlist
   
    j = 6000
   
    for i, vec in enumerate(n):
        M = Mlist[i]
        Mlist.append(M)
        if i not in ind: continue
        vec = vec/np.linalg.norm(vec)
        C21 = LammpsAtom(2,   j, shifted+vec*29.5, [M]) #25.5 radius, 4 A bond length
        C22 = LammpsAtom(2,   j, shifted+vec*32.0, [C21]) #2.5 A bond length
        C23 = LammpsAtom(2, j+i+2, shifted+vec*34.5, [C22]) #2.5 A bond length
        C31 = LammpsAtom(3, j+i+2, shifted+vec*37.5, [C23]) #3.0 A bond length
        C32 = LammpsAtom(3, j+i+2, shifted+vec*41.0, [C31]) #3.5 A bond length
        P31 = LammpsAtom(4, j+i+2, shifted+vec*43.0, [C32]) #2.0 A bond length
        P32 = LammpsAtom(4, j+i+2, shifted+vec*(43.0+1.866), [P31]) # must be accurate A bond length
        O01 = LammpsAtom(5, j+i+2, shifted+vec*(43.0+1.853+1.866), [P32]) # must be accurate A bond length
        A,B,P = vec*43.0, vec*(43.0+1.866),vec*(43.0+1.853+1.866)
        C = computeCoordinateMissingICSolve(B,A,B, 1.853,180-100,0)
        D = computeCoordinateMissingIC(C, A, B, 1.853, 180-100, 180)
        O02 = LammpsAtom(8, j+i+2, shifted+C, [P32])
        O03 = LammpsAtom(8, j+i+2, shifted+D, [P32])
        P32.bonded = [P31, O01, O02, O03] # A fixer
        v1,v2,v3 = vec, (C-B)/np.linalg.norm(C-B), (D-B)/np.linalg.norm(D-B)
        lammps_data.content += [C21, C22, C23, C31, C32, P31, P32, O01, O02, O03]
       
        # write branch 1
        C211 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*2.0, [O01])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*4.5, [C211])
        C311 = LammpsAtom(3, j+len(n)+2+3*i+0, shifted+P+v1*7.5, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*10.0, [C311])
        C214 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*12.5, [C213])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+0, shifted+P+v1*14.0, [C214])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+0, shifted+P+v1*16.0, [O011])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+0, shifted+P+v1*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+0, shifted+P+v1*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, C212, C311, C213, C214, O011, P311, C611, P312]

        # write branch 2
        C211 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*2.0, [O02])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*4.5, [C211])
        C311 = LammpsAtom(3, j+len(n)+2+3*i+1, shifted+C+v2*7.5, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*10.0, [C311])
        C214 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*12.5, [C213])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+1, shifted+C+v2*14.0, [C214])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+1, shifted+C+v2*16.0, [O011])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+1, shifted+C+v2*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+1, shifted+C+v2*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, C212, C311, C213, C214, O011, P311, C611, P312]
        
        # write branch 3
        C211 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*2.0, [O03])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*4.5, [C211])
        C311 = LammpsAtom(3, j+len(n)+2+3*i+2, shifted+D+v3*7.5, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*10.0, [C311])
        C214 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*12.5, [C213])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+2, shifted+D+v3*14.0, [C214])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+2, shifted+D+v3*16.0, [O011])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+2, shifted+D+v3*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+2, shifted+D+v3*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, C212, C311, C213, C214, O011, P311, C611, P312]

   
    j = 8000
    # move to a second NC-L2  
   
    #shifted = np.array([1/2*L,1/2*L,0]) #30
    shifted = np.array([1/2*L, np.sqrt(3)*1/6*L, L*np.sqrt(6)/3])
    Mlist = []
    d = computeEuclideanDistancesMatrix(n)
    d = np.sqrt(d)
    d[d==0] = 100
    index = np.argwhere(d < 0.6)
    for i, loc in enumerate(n):
        Mlist.append(LammpsAtom(1, j, shifted+loc*10, []))
    for i, j in index:
        Mlist[i].bonded += [Mlist[j]]
    lammps_data.content += Mlist
    
    j = 8000
   
    for i, vec in enumerate(n):
        M = Mlist[i]
        Mlist.append(M)
        if i not in ind: continue
        vec = vec/np.linalg.norm(vec)
        C21 = LammpsAtom(2,   j, shifted+vec*29.5, [M]) #25.5 radius, 4 A bond length
        C22 = LammpsAtom(2,   j, shifted+vec*32.0, [C21]) #2.5 A bond length
        C23 = LammpsAtom(2, j+i+2, shifted+vec*34.5, [C22]) #2.5 A bond length
        C31 = LammpsAtom(3, j+i+2, shifted+vec*37.5, [C23]) #3.0 A bond length
        C32 = LammpsAtom(3, j+i+2, shifted+vec*41.0, [C31]) #3.5 A bond length
        P31 = LammpsAtom(4, j+i+2, shifted+vec*43.0, [C32]) #2.0 A bond length
        P32 = LammpsAtom(4, j+i+2, shifted+vec*(43.0+1.866), [P31]) # must be accurate A bond length
        O01 = LammpsAtom(5, j+i+2, shifted+vec*(43.0+1.853+1.866), [P32]) # must be accurate A bond length
        A,B,P = vec*43.0, vec*(43.0+1.866),vec*(43.0+1.853+1.866)
        C = computeCoordinateMissingICSolve(B,A,B, 1.853,180-100,0)
        D = computeCoordinateMissingIC(C, A, B, 1.853, 180-100, 180)
        O02 = LammpsAtom(8, j+i+2, shifted+C, [P32])
        O03 = LammpsAtom(8, j+i+2, shifted+D, [P32])
        P32.bonded = [P31, O01, O02, O03] # A fixer
        v1,v2,v3 = vec, (C-B)/np.linalg.norm(C-B), (D-B)/np.linalg.norm(D-B)
        lammps_data.content += [C21, C22, C23, C31, C32, P31, P32, O01, O02, O03]
       
        # write branch 1
        C211 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*2.0, [O01])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*4.5, [C211])
        C311 = LammpsAtom(3, j+len(n)+2+3*i+0, shifted+P+v1*7.5, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*10.0, [C311])
        C214 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*12.5, [C213])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+0, shifted+P+v1*14.0, [C214])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+0, shifted+P+v1*16.0, [O011])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+0, shifted+P+v1*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+0, shifted+P+v1*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, C212, C311, C213, C214, O011, P311, C611, P312]

        # write branch 2
        C211 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*2.0, [O02])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*4.5, [C211])
        C311 = LammpsAtom(3, j+len(n)+2+3*i+1, shifted+C+v2*7.5, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*10.0, [C311])
        C214 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*12.5, [C213])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+1, shifted+C+v2*14.0, [C214])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+1, shifted+C+v2*16.0, [O011])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+1, shifted+C+v2*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+1, shifted+C+v2*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, C212, C311, C213, C214, O011, P311, C611, P312]
        
        # write branch 3
        C211 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*2.0, [O03])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*4.5, [C211])
        C311 = LammpsAtom(3, j+len(n)+2+3*i+2, shifted+D+v3*7.5, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*10.0, [C311])
        C214 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*12.5, [C213])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+2, shifted+D+v3*14.0, [C214])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+2, shifted+D+v3*16.0, [O011])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+2, shifted+D+v3*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+2, shifted+D+v3*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, C212, C311, C213, C214, O011, P311, C611, P312]
        
   
    j = 10000
    # move to a second NC-L2  
   
    #shifted = np.array([1/2*L,1/2*L,0]) #30
    shifted = np.array([0, np.sqrt(3)*2/3*L, L*np.sqrt(6)/3])
    Mlist = []
    d = computeEuclideanDistancesMatrix(n)
    d = np.sqrt(d)
    d[d==0] = 100
    index = np.argwhere(d < 0.6)
    for i, loc in enumerate(n):
        Mlist.append(LammpsAtom(1, j, shifted+loc*10, []))
    for i, j in index:
        Mlist[i].bonded += [Mlist[j]]
    lammps_data.content += Mlist
    
    j = 10000   
   
    for i, vec in enumerate(n):
        M = Mlist[i]
        Mlist.append(M)
        if i not in ind: continue
        vec = vec/np.linalg.norm(vec)
        C21 = LammpsAtom(2,   j, shifted+vec*29.5, [M]) #25.5 radius, 4 A bond length
        C22 = LammpsAtom(2,   j, shifted+vec*32.0, [C21]) #2.5 A bond length
        C23 = LammpsAtom(2, j+i+2, shifted+vec*34.5, [C22]) #2.5 A bond length
        C31 = LammpsAtom(3, j+i+2, shifted+vec*37.5, [C23]) #3.0 A bond length
        C32 = LammpsAtom(3, j+i+2, shifted+vec*41.0, [C31]) #3.5 A bond length
        P31 = LammpsAtom(4, j+i+2, shifted+vec*43.0, [C32]) #2.0 A bond length
        P32 = LammpsAtom(4, j+i+2, shifted+vec*(43.0+1.866), [P31]) # must be accurate A bond length
        O01 = LammpsAtom(5, j+i+2, shifted+vec*(43.0+1.853+1.866), [P32]) # must be accurate A bond length
        A,B,P = vec*43.0, vec*(43.0+1.866),vec*(43.0+1.853+1.866)
        C = computeCoordinateMissingICSolve(B,A,B, 1.853,180-100,0)
        D = computeCoordinateMissingIC(C, A, B, 1.853, 180-100, 180)
        O02 = LammpsAtom(8, j+i+2, shifted+C, [P32])
        O03 = LammpsAtom(8, j+i+2, shifted+D, [P32])
        P32.bonded = [P31, O01, O02, O03] # A fixer
        v1,v2,v3 = vec, (C-B)/np.linalg.norm(C-B), (D-B)/np.linalg.norm(D-B)
        lammps_data.content += [C21, C22, C23, C31, C32, P31, P32, O01, O02, O03]
       
        # write branch 1
        C211 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*2.0, [O01])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*4.5, [C211])
        C311 = LammpsAtom(3, j+len(n)+2+3*i+0, shifted+P+v1*7.5, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*10.0, [C311])
        C214 = LammpsAtom(2, j+len(n)+2+3*i+0, shifted+P+v1*12.5, [C213])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+0, shifted+P+v1*14.0, [C214])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+0, shifted+P+v1*16.0, [O011])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+0, shifted+P+v1*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+0, shifted+P+v1*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, C212, C311, C213, C214, O011, P311, C611, P312]

        # write branch 2
        C211 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*2.0, [O02])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*4.5, [C211])
        C311 = LammpsAtom(3, j+len(n)+2+3*i+1, shifted+C+v2*7.5, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*10.0, [C311])
        C214 = LammpsAtom(2, j+len(n)+2+3*i+1, shifted+C+v2*12.5, [C213])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+1, shifted+C+v2*14.0, [C214])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+1, shifted+C+v2*16.0, [O011])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+1, shifted+C+v2*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+1, shifted+C+v2*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, C212, C311, C213, C214, O011, P311, C611, P312]
        
        # write branch 3
        C211 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*2.0, [O03])
        C212 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*4.5, [C211])
        C311 = LammpsAtom(3, j+len(n)+2+3*i+2, shifted+D+v3*7.5, [C212])
        C213 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*10.0, [C311])
        C214 = LammpsAtom(2, j+len(n)+2+3*i+2, shifted+D+v3*12.5, [C213])
        O011 = LammpsAtom(5, j+len(n)+2+3*i+2, shifted+D+v3*14.0, [C214])
        P311 = LammpsAtom(4, j+len(n)+2+3*i+2, shifted+D+v3*16.0, [O011])
        C611 = LammpsAtom(6, j+len(n)+2+3*i+2, shifted+D+v3*(16.0+3.08), [P311])
        P312 = LammpsAtom(4, j+len(n)+2+3*i+2, shifted+D+v3*(16.0+3.08*2), [C611])
        lammps_data.content += [C211, C212, C311, C213, C214, O011, P311, C611, P312]


    # This part is solvation
    j = 12000
    nx,ny,nz = 25, 44, 20
    spacing_x, spacing_y, spacing_z = 5, 5, 11
    for mol_id in range(nx*ny*nz):
        ix, iy, iz = mol_id % nx, (mol_id // nx) % ny, (mol_id // (nx * ny)) % nz       
        x, y, z = ix * spacing_x, iy * spacing_y, iz * spacing_z
        
        CN1 = LammpsAtom(7, mol_id+j, [x,y,z+200+ 0.0], None)
        P31 = LammpsAtom(4, mol_id+j, [x,y,z+200+ 2.5], [CN1])
        C61 = LammpsAtom(9, mol_id+j, [x,y,z+200+ 5.5], [P31])
        P32 = LammpsAtom(4, mol_id+j, [x,y,z+200+ 8.5], [C61])
        C31 = LammpsAtom(3, mol_id+j, [x,y+2, z+200+9], [P32])
        C21 = LammpsAtom(2, mol_id+j, [x,y+4, z+200+9], [C31])
        
        lammps_data.content += [CN1, P31, C61, P32, C31, C21]

    

    lammps_data.build_atoms(mass_dict, pair_dict)
    lammps_data.build_bonds(bond_dict)
    lammps_data.build_angles(angle_dict)
    lammps_data.output_file('L2-111Surface-Reduced.lammpsdata')