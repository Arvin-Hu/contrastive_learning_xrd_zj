import re

class ElementEncoder:
    def __init__(self):
        # 预定义元素周期表中的所有元素符号
        self.all_elements = [
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
            'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
            'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
            'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
        ]
        
        # 创建一个元素到编码的映射
        self.element_to_code = {element: i+1 for i, element in enumerate(self.all_elements)}
        self.code_to_element = {i+1: element for i, element in enumerate(self.all_elements)}
    
    def extract_elements(self, formula):
        """
        从化学式中提取元素种类
        
        参数:
        formula (str): 化学式字符串，如 "H2O", "C6H12O6", "NaCl"
        
        返回:
        list: 元素符号列表
        """
        # 正则表达式匹配元素符号（大写字母+可选小写字母）
        pattern = r'[A-Z][a-z]?'
        elements = re.findall(pattern, formula)
        
        # 去重并保持顺序
        unique_elements = []
        for element in elements:
            # 验证是否为有效元素符号
            if element in self.all_elements and element not in unique_elements:
                unique_elements.append(element)
        
        return unique_elements
    
    def encode_elements(self, elements):
        """
        对元素列表进行编码
        
        参数:
        elements (list): 元素符号列表
        
        返回:
        codes (list): int编码列表，包含编码后的元素
        """
        codes = [self.element_to_code[element] for element in elements] 
        return codes
    
    def decode_elements(self, codes):
        """
        将编码解码回元素符号
        
        参数:
        codes (list): 编码列表
        
        返回:
        list: 元素符号列表
        """
        elements = []
        for code in codes:
            if code in self.code_to_element:
                elements.append(self.code_to_element[code])
        return elements
    
    def encode_formula(self, formula):
        """
        对化学式进行编码
        
        参数:
        formula (str): 化学式字符串，如 "H2O", "C6H12O6", "NaCl"
        
        返回:
        codes (list): int编码列表，包含编码后的元素
        """
        elements = self.extract_elements(formula)
        codes = self.encode_elements(elements)
        return codes
    def get_formula_info(self, formula):
        """
        获取化学式的完整信息
        
        参数:
        formula (str): 化学式
        
        返回:
        dict: 包含提取的元素、编码等信息的字典
        """
        # 提取元素
        elements = self.extract_elements(formula)
        
        # 编码元素
        codes = self.encode_elements(elements)
        
        # 创建结果字典
        result = {
            'formula': formula,
            'elements': elements,
            'element_count': len(elements),
            'codes': codes,
        }
        
        return result
    
    def get_one_hot_encoding(self, elements=None, formula=None):
        """
        获取元素的onehot编码
        
        参数:
        elements (list, optional): 元素列表
        formula (str, optional): 化学式字符串
        
        返回:
        dict: 元素的onehot编码
        """
        if formula:
            elements = self.extract_elements(formula)
        elif not elements:
            raise ValueError("必须提供elements或formula参数")
        
        # 创建onehot编码
        encoding = [0] * len(self.all_elements)
        for element in elements:
            if element in self.element_to_code:
                encoding[self.element_to_code[element]] = 1
        
        return encoding


# 使用示例
if __name__ == "__main__":
    # 创建编码器实例
    encoder = ElementEncoder()
    
    # 示例化学式
    test_formulas = ["H2O", "C6H12O6", "NaCl", "CH3COOH", "Fe2O3"]
    
    print("化学式元素提取和编码示例：")
    print("=" * 50)
    
    for formula in test_formulas:
        print(f"\n化学式: {formula}")
        
        # 获取化学式信息
        info = encoder.get_formula_info(formula)
        
        print(f"提取的元素: {info['elements']}")
        print(f"元素数量: {info['element_count']}")
        print(f"元素编码: {info['codes']}")
        
        # 获取一热编码
        one_hot = encoder.get_one_hot_encoding(formula=formula)
        print(f"{formula}的onehot编码: {one_hot}")
        
        print("-" * 50)
    
