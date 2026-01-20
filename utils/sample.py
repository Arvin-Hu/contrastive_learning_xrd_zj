import json
import re

SPACE_GROUP_DICT = {
    'P1', 'P-1', 'P2', 'P2_1', 'C2', 'Pm', 'Cm', 'Pc', 'Cc',
    'P2/m', 'P2_1/m', 'C2/m', 'P2/c', 'P2_1/c', 'C2/c',
    'P222', 'P222_1', 'P2_12_12', 'P2_12_12_1', 'C222', 'C222_1',
    'F222', 'I222', 'I2_12_12_1', 'Pmm2', 'Pmc2_1', 'Pcc2',
    'Pma2', 'Pca2_1', 'Pnc2', 'Pmn2_1', 'Pba2', 'Pna2_1',
    'Pnn2', 'Cmm2', 'Cmc2_1', 'Ccc2', 'Amm2', 'Abm2', 'Ama2',
    'Aba2', 'Fmm2', 'Fdd2', 'Imm2', 'Iba2', 'Ima2',
    'Pmmm', 'Pnnn', 'Pccm', 'Pban', 'Pmma', 'Pnna', 'Pmna',
    'Pcca', 'Pbam', 'Pccn', 'Pbcm', 'Pnnm', 'Pmmn',
    'Pbcn', 'Pbca', 'Pnma', 'Cmcm', 'Cmca', 'Cmmm', 'Cmce', 'Cmme',
    'Cccm', 'Cmma', 'Ccca', 'Fmmm', 'Fddd', 'Immm', 'Ccce',
    'Ibam', 'Ibca', 'Imma', 'P4', 'P4_1', 'P4_2', 'P4_3',
    'I4', 'I4_1', 'P-4', 'I-4', 'P4/m', 'P4_2/m', 'P4/n',
    'P4_2/n', 'I4/m', 'I4_1/a', 'P422', 'P42_12', 'P4_122',
    'P4_12_12', 'P4_222', 'P4_22_12', 'P4_322', 'P4_32_12',
    'I422', 'I4_122', 'P4mm', 'P4bm', 'P4_2cm', 'P4_2nm',
    'P4cc', 'P4nc', 'P4_2mc', 'P4_2bc', 'I4mm', 'I4cm',
    'I4_1md', 'I4_1cd', 'P-42m', 'P-42c', 'P-42_1m', 'P-42_1c',
    'P-4m2', 'P-4c2', 'P-4b2', 'P-4n2', 'I-4m2', 'I-4c2',
    'I-42m', 'I-42d', 'P4/mmm', 'P4/mcc', 'P4/nbm', 'P4/nnc',
    'P4/mbm', 'P4/mnc', 'P4/nmm', 'P4/ncc', 'P4_2/mmc',
    'P4_2/mcm', 'P4_2/nbc', 'P4_2/nnm', 'P4_2/mbc', 'P4_2/mnm',
    'P4_2/nmc', 'P4_2/ncm', 'I4/mmm', 'I4/mcm', 'I4_1/amd',
    'I4_1/acd', 'P3', 'P3_1', 'P3_2', 'R3', 'P-3', 'R-3',
    'P312', 'P321', 'P3_112', 'P3_121', 'P3_212', 'P3_221',
    'R32', 'P3m1', 'P31m', 'P3c1', 'P31c', 'R3m', 'R3c',
    'P-31m', 'P-31c', 'P-3m1', 'P-3c1', 'R-3m', 'R-3c',
    'P6', 'P6_1', 'P6_5', 'P6_2', 'P6_4', 'P6_3',
    'P-6', 'P6/m', 'P6_3/m', 'P622', 'P6_122', 'P6_522',
    'P6_222', 'P6_422', 'P6_322', 'P6mm', 'P6cc', 'P6_3cm',
    'P6_3mc', 'P-6m2', 'P-6c2', 'P-62m', 'P-62c', 'P6/mmm',
    'P6/mcc', 'P6_3/mcm', 'P6_3/mmc', 'P23', 'F23', 'I23',
    'P2_13', 'I2_13', 'Pm-3', 'Pn-3', 'Fm-3', 'Fd-3', 'Im-3',
    'Pa-3', 'Ia-3', 'P432', 'P4_232', 'F432', 'F4_132',
    'I432', 'P4_332', 'P4_132', 'I4_132', 'P-43m', 'F-43m',
    'I-43m', 'P-43n', 'F-43c', 'I-43d', 'Pm-3m', 'Pn-3n',
    'Pm-3n', 'Pn-3m', 'Fm-3m', 'Fm-3c', 'Fd-3m', 'Fd-3c',
    'Im-3m', 'Ia-3d', 'Aea2', 'Aem2'
}
def get_crystal_system(s):
    return s.replace('This material belongs to', '').replace('system.', '').strip()



def extract_space_group(text):
    text = text.replace('.', '')
    words = re.findall(r'\b([A-Za-z\d_/\-]+)\b', text)
    found = []
    for word in words:
        # 精确匹配词典中的空间群
        if word in SPACE_GROUP_DICT:
            found.append(word)
            break
        # 检查变体（如大小写不敏感）
        elif word.upper() in [sg.upper() for sg in SPACE_GROUP_DICT]:
            # 找到对应的大小写正确版本
            for sg in SPACE_GROUP_DICT:
                if sg.upper() == word.upper():
                    found.append(sg)
                    break
    if len(found) == 0:
        raise ValueError("No space group in the text: {}".format(text))
    
    return found[0]
def gen_sample(bandgap_input_jsonl, crystal_system_jsonl, space_group_jsonl, formation_energy_jsonl, formula_jsonl, lattice_jsonl, output_jsonl):
    pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    samples = {}
    with open(bandgap_input_jsonl, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            sample = {'xrd': data['xrd'], 'bandgap': float(re.findall(pattern, data["conversations"][1]["value"])[0])}
            samples[data['xrd']] = sample
            
    with open(crystal_system_jsonl, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            samples[data['xrd']].update({'crystal_system': get_crystal_system(data["conversations"][1]["value"])})
            
    with open(space_group_jsonl, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            samples[data['xrd']].update({'space_group': extract_space_group(data["conversations"][1]["value"])})
            
    with open(formation_energy_jsonl, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            samples[data['xrd']].update({'formation_energy': float(re.findall(pattern, data["conversations"][1]["value"])[0])})
            
    with open(formula_jsonl, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            samples[data['xrd']].update({'formula': data["conversations"][1]["value"].strip()})
            
    with open(lattice_jsonl, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            samples[data['xrd']].update({'lattice_parameters': [float(x) for x in re.findall(pattern, data["conversations"][1]["value"])]})
            
    with open(output_jsonl, "w", encoding='utf-8') as out_f: 
        for key in samples:
            out_f.write(json.dumps(samples[key]) + "\n")
            
            
if __name__ == '__main__':
    bandgap_jsonl = '/mnt/minio/battery/xrd/datasets/MP_bandgap-QA-train.jsonl'
    crystal_system_jsonl = '/mnt/minio/battery/xrd/datasets/MP_crystalsystem_QA_train.jsonl'
    space_group_jsonl = '/mnt/minio/battery/xrd/datasets/MP_spacegroup-QA-train.jsonl'
    formation_energy_jsonl = '/mnt/minio/battery/xrd/datasets/MP_formationenergy-QA-train.jsonl'
    formula_jsonl = '/mnt/minio/battery/xrd/datasets/MP_formula-simple-QA-train.jsonl'
    lattice_jsonl = '/mnt/minio/battery/xrd/datasets/MP_latticeparameters-QA-train.jsonl'
    output_jsonl = '/mnt/minio/battery/xrd/datasets/MP_xrd-train.jsonl'
    gen_sample(bandgap_jsonl, crystal_system_jsonl, space_group_jsonl, formation_energy_jsonl, formula_jsonl, lattice_jsonl, output_jsonl)