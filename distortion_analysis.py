import os
import yaml
import traceback
import numpy as np
import pandas as pd
import shutil
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from workflow_utils import sort_sites

POLAR_POINT_GROUPS = ['C1','C2','C3','C4','C6', 'm','mm2','3m','4mm','6mm']


def calc_point_groups(dim_str, kpt, index, amplitude=5):
    """Calculate the point group of a distortion mode using phonopy
    Args:
        dim_str (str): Dimension of the supercell (e.g. '2 2 2')
        kpt (str): k-point of the distortion mode (e.g. '0 0 0')
        index (int): Index of the distortion mode        
    """
    ### Generate distorted example structure
    with open('mod.conf', 'w+') as f:
        f.write(f"DIM = {dim_str}\nMODULATION = {dim_str}, {kpt} {index} {amplitude}")
    os.system(f'phonopy -q mod.conf')
    
    ### Calculate its point group
    os.system(f'phonopy -q --symmetry -c MPOSCAR > sym.txt')
    with open('sym.txt') as f:
        lines = f.readlines()
        return lines[3].split()[-1].strip("'")


def parse_irreps(phon_dir, irreps_filepath, dim_str='2 2 2', kpt='0 0 0'):
    """Parse the irreps of the normal modes of a distorted structure
    Args:
        phon_dir (str): Subdirectory containing the phonopy output files (irreps.yaml, POSCAR, etc.)
        irreps_filepath (str): Path to file containing the irreps of the normal modes
        dim_str (str): Dimension of the supercell (e.g. '2 2 2')
        kpt (str): k-point of the distortion mode (e.g. '0 0 0')
    Returns:
        df (pd.DataFrame): DataFrame containing the irreps of the normal modes
    """
    base_dir = os.getcwd()
    os.chdir(phon_dir)
    
    try:
        ### Parse irrep details
        with open(irreps_filepath) as f:
            irreps = yaml.safe_load(f)
            df = pd.DataFrame(irreps['normal_modes'])

        ### Find inversion (not yet tested)
        # df['centrosym'] = [table[1][1] == 0 for table in df['characters']]

        ### Find broken mirror planes (not yet tested)
        # df['broken_mirrors'] = [detect_broken_mirrors(table) for table in df['characters']]

        ### Find point group of each distortion mode
        indices = [ind[0] for ind in df['band_indices']]
        df['point_group'] = [calc_point_groups(dim_str=dim_str, kpt=kpt, index=index) for index in indices]
        
        ### Find polar modes
        df['is_polar'] = [pg in POLAR_POINT_GROUPS for pg in df['point_group']]

        ### Find potential ferroelectric modes
        df['potential_fe'] = [is_pol and frequency < -0.05 for is_pol, frequency in zip(df['is_polar'], df['frequency'])]

        ### Find multiplicity
        df['multiplicity'] = [len(ind) for ind in df['band_indices']]

        os.chdir(base_dir) 

    ### If something breaks, print error, get back to original directory
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        os.chdir(base_dir) 
        return None
    
    return df


def gen_mod_struc(phon_path, labels, kpts, indices, disps, moddims, symprec=1E-4):
    """Generate a single modulated structure from a directory containing a 
    complete phonopy calculation.
    Args:
        phon_path (str): Path to phonopy directory
        labels (list): List of labels (str) for each modulation, 'e.g. ['G', 'M']
        kpts (list): List of k-points (str) for each modulation, e.g. ['0 0 0', '0.5 0.5 0']
        indices (list): List of indices (int) for each modulation
        disps (list): List of displacements (float) for each modulation
        moddims (list): List of modulation dimensions, e.g. [1,2,2]
    Returns:
        struc (Structure): Modulated structure
        (str): Path to modulated structure directory
    """
    ### Create directory structure for modulated structures
    og_dir = os.getcwd()
    mods = [f'{label}{index}' for label,index in zip(labels,indices)]
    moddir = '/'.join(phon_path.split('/')[:-1] + ['modulation'])
    phon_tag = phon_path.split('/')[-1] 
    subdir = f'mod_{phon_tag}_{"_".join(mods)}'
    os.makedirs(moddir, exist_ok=True)
    os.chdir(moddir)
    os.makedirs(subdir, exist_ok=True)
    os.chdir(subdir)

    ### Create modulation config file
    with open('mod.conf', 'w+') as f:
        shutil.copy(f'{phon_path}/phonopy_disp.yaml', '.')
        shutil.copy(f'{phon_path}/FORCE_SETS', '.')
        f.write('FC_SYMMETRY = .TRUE.\n')
        moddim_str = " ".join([str(i) for i in moddims])
        mod_line = f'MODULATION = {moddim_str}'
        for kpt, index, disp in zip(kpts, indices, disps):
            mod_line += f', {kpt} {index} {disp}'
        f.write(mod_line)
    
    ### Generate modulated structures
    os.system(f'phonopy mod.conf')

    ### Generate, reduce, and sort the modulated structures
    struc = Structure.from_file('MPOSCAR')
    sga = SpacegroupAnalyzer(struc, symprec=symprec)
    print(f'{subdir},  SG = {sga.get_space_group_symbol()} ({sga.get_space_group_number()})')
    struc = sga.get_primitive_standard_structure()
    struc = sort_sites(struc)
    os.chdir(og_dir)
    return struc, f'{moddir}/{subdir}'


def sort_sites(struc, unwrap_tol=0.05, verbose=False):
    """Sort sites of a pymatgen structure by coordinates and species. Useful 
    for ensuring distorted structure variants are comparable/interpolatable.
    Args:
        struc (Structure): structure
        unwrap_tol (float): maximum distance from upper edge of unit cell
                            below which sites will be unwrapped to negative 
                            coordinates (fractional units) 
    Returns:
        Structure: sorted structure
    """
    coords = np.array(struc.frac_coords)

    ### Wrap very high coords to negative to avoid sort errors
    for i in range(coords.shape[0]):
        for j in range(coords.shape[1]):
            if coords[i,j] + unwrap_tol >= 1:
                if verbose: print(f'sort_sites: at edge: {coords[i,j]}')
                coords[i,j] = coords[i,j] - 1 
                if verbose: print(f'sort_sites: wrapped: {coords[i,j]}')
    species = np.array(struc.species)
    sorted_indices = np.lexsort((coords[:,0], coords[:,1], coords[:,2], species))
    return Structure.from_sites([struc[i] for i in sorted_indices])
