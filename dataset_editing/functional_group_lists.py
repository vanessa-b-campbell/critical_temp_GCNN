full_func_stats = []

# add every element in the periodic table at some point

# all pattern lists
phenol_list = []
benzene_list = []
carboxylic_list = []
fluorine_list = []
nitrogen_list = []
silicon_list = []
xe_list = []
chlorine_list = []
titanium_list = []
argon_list = []
bromine_list = []
sulfur_list = []
iodine_list = []
toluene_list = []
aniline_list = []
acetophenone_list = []
benzaldehyde_list=[]
benzoic_acid_list = []
benzonitrile_list = []
ortho_xylene_list = []
styrene_list = []
oxygen_list = []
neon_list = []
krypton_list =[]
radon_list = []
helium_list = []
phosphorus_list=[]
arsenic_list = []
antimony_list = []
carbon_list = []
selenium_list = []
caesium_list = []
germanium_list = []
tellurium_list = []
tin_list = []
thallium_list = []
hydrogen_list =[]


# SMILES pattern list
# ordered loosely on periodic table columns and aromaic groups

# [  'SMILES string of functioanl group'  , empty_group_list,    'name of the functional group'   ]

func_group_list = [ ['C1=CC=CC=C1', benzene_list, 'benzene'], 
                    ['C(=O)O', carboxylic_list, 'carboxylic acid'], ['[Ti]', thallium_list, 'thallium'],   

                ['N', nitrogen_list, 'nitrogen'], ['P', phosphorus_list, 'phosphorus'],
                

                
                ['S', sulfur_list, 'sulfur'], ['O', oxygen_list, 'oxygen' ],
                

                ['Cl', chlorine_list, 'chlorine'], ['F', fluorine_list, 'fluorine'],  
                ['I', iodine_list, 'iodine'], ['Br', bromine_list, 'bromine'],

                ['C1=CC=C(C=C1)O', phenol_list, 'phenol'],['Cc1ccccc1', toluene_list, 'toluene'], 
                ['Nc1ccccc1 c1ccc(cc1)N', aniline_list, 'aniline'], 
                ['O=C(c1ccccc1)C CC(=O)c1ccccc1', acetophenone_list, 'acetophenone'],
                ['O=Cc1ccccc1 c1ccc(cc1)C=O', benzaldehyde_list, 'benzaldehyde'],
                ['C1=CC=C(C=C1)C(=O)O', benzoic_acid_list, 'benzoic acid'],
                ['C1=CC=C(C=C1)C#N', benzonitrile_list, 'benzonitrile'],
                ['CC1=CC=CC=C1C', ortho_xylene_list, 'ortho-xylene'],
                ['C=CC1=CC=CC=C1', styrene_list, 'styrene'], 

                ['[Ne]', neon_list, 'neon'], ['[Kr]', krypton_list, 'krypton'], 
                ['[Rn]', radon_list, 'radon'], ['[He]', helium_list, 'helium'],
                ['[Ar]', argon_list, 'Argon'], ['[Xe]', xe_list, 'Xe'],

                ['[Cs]', caesium_list, 'caesium'],['C', carbon_list, 'carbon']]


# second list for SMARTS patterns
# mostly for metalloids/ metals/ hydrogen
# [  '[SMARTS string]',   empty_group_list,    'name of functional group' ]
metalloid_group_list = [["[Si]", silicon_list, 'silicon'],['[Se]', selenium_list, 'selenium'], 
                        ['[As]', arsenic_list, 'arsenic'],['[Ge]', germanium_list, 'germanium'],
                        ['[Sn]', tin_list, 'tin'], ['[Ti]',titanium_list, 'titanium'],
                        ['[H]', hydrogen_list, 'hydrogen']]

