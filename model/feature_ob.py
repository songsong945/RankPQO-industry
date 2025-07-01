import json
import re
from abc import ABCMeta, abstractmethod

import numpy as np
import torch

# OceanBase 目前只跑 pred 模式，所以只保留必要的 FEATURE_LIST
FEATURE_LIST = ['Node Type', 'Plan Rows', 'Plan Width']
LABEL_LIST = ['Actual Startup Time', 'Actual Total Time', 'Actual Self Time']

UNKNOWN_OP_TYPE = "Unknown"
# 根据 OceanBase EXPLAIN JSON 中的 OPERATOR 字段精确映射
SCAN_TYPES = ["TABLE FULL SCAN"]
JOIN_TYPES = ["HASH JOIN", "MERGE JOIN", "NESTED LOOP", "HASH RIGHT SEMI JOIN"]
OP_TYPES = [UNKNOWN_OP_TYPE] + SCAN_TYPES + JOIN_TYPES

OPERATORS = ['!~~', '~~', '<=', '>=', '=', '<', '>']
UNKNOWN_COLUMN = 'Unknown'
# 只保留 Unknown，后面不会用到列编码
COLUMNS = [UNKNOWN_COLUMN]


def op_to_one_hot(op_name: str) -> np.ndarray:
    """OceanBase 算子名称的 one-hot 编码"""
    vec = np.zeros(len(OP_TYPES), dtype=int)
    idx = OP_TYPES.index(op_name) if op_name in OP_TYPES else 0
    vec[idx] = 1
    return vec


def preprocess_z(vector: list,
                 params: list,
                 preprocessing_infos: list) -> torch.Tensor:
    """安全版本的参数预处理，缺少 distinct_values 时跳过 embedding"""
    processed = []
    # 转置，方便按参数维度展开
    vector = list(zip(*vector))
    for param, prep in zip(params, preprocessing_infos):
        dtype = param["data_type"]
        ptype = prep["type"]
        layer = vector.pop(0)
        if dtype == "float" and ptype == "std_normalization":
            mean = prep["mean"]
            std = torch.sqrt(prep["variance"])
            processed.append((np.array(layer).astype(float) - mean) / std)
        elif dtype == "int" and ptype == "embedding":
            distinct = param.get("distinct_values", [])
            vocab = {w: i for i, w in enumerate(distinct)}
            arr = np.array([vocab.get(v, len(vocab)) for v in layer])
            processed.append(arr)
        elif dtype == "text":
            distinct = param.get("distinct_values", [])
            vocab = {w: i for i, w in enumerate(distinct)}
            arr = np.array([vocab.get(v, len(vocab)) for v in layer])
            processed.append(arr)
        else:
            # 其他情况，跳过
            continue
    # 转回 (batch, max_len)
    return torch.tensor(np.transpose(np.array(processed)))


def clean_column_name(raw: str) -> str:
    s = re.sub(r'::[a-zA-Z_]+', '', raw)
    return re.sub(r'^\W+|\W+$', '', s)


def extract_predicates_from_condition(condition: str, alias: str):
    preds = []
    for op in OPERATORS:
        if op in condition:
            parts = re.split(r'\s+AND\s+|\s+OR\s+', condition)
            for part in parts:
                if op in part:
                    l, r = map(str.strip, re.split(re.escape(op), part, maxsplit=1))
                    lc = clean_column_name(l)
                    rc = clean_column_name(r)
                    full = f"{alias}.{lc}"
                    # 排除列与列的比较
                    if not (re.fullmatch(r'[a-z_]+\.[a-z_]+', lc, re.I)
                            and re.fullmatch(r'[a-z_]+\.[a-z_]+', rc, re.I)):
                        preds.append((full, op))
    return preds


def extract_predicates_from_node(plan: dict):
    # OceanBase JSON 里暂时没有 Filter/Cond 字段
    return []


class SampleEntity:
    """简化版本，只用于 pred 模式下，提供固定长度的 get_feature()"""
    def __init__(self,
                 node_type: np.ndarray,
                 width: int,
                 pred_encoding: np.ndarray,
                 left, right):
        self.node_type = node_type
        self.width = width
        self.pred_encoding = pred_encoding
        self.left = left
        self.right = right

    def get_feature(self) -> np.ndarray:
        # [one_hot(op)] + [width] + [pred_encoding]
        return np.hstack((self.node_type,
                          np.array([self.width], dtype=int),
                          self.pred_encoding))
        
    def get_feature_len(self) -> int:
        # 返回特征向量的长度，供 model.fit_with_test 使用
        return self.get_feature().shape[0]
    
    def get_left(self):
        return self.left

    def get_right(self):
        return self.right


class FeatureParser(metaclass=ABCMeta):
    @abstractmethod
    def extract_pred_feature(self, node: dict) -> SampleEntity:
        pass

    @abstractmethod
    def extract_feature(self, node: dict) -> SampleEntity:
        pass


class OBAnalyzeJsonParser(FeatureParser):
    """OceanBase EXPLAIN FORMAT=JSON 的解析器，同时支持 est 和 pred 特征"""
    def __init__(self,
                 input_relations: list,
                 columns: list,
                 max_pred_len: int = 30):
        # input_relations 仅用来定死 encoded_input_tables 的长度（这里我们不再 use）
        self.input_relations = input_relations
        self.columns = columns
        self.max_pred_len = max_pred_len

    def _get_children(self, node: dict):
        ch = []
        for k in ("CHILD_1", "CHILD_2"):
            if k in node and isinstance(node[k], dict):
                ch.append(node[k])
        return ch

    def _parse_output_width(self, output_str: str) -> int:
        # 例："output([a], [b], [c])" → 3
        m = re.search(r'\[\s*(.*?)\s*\]', output_str or '')
        if not m:
            return 0
        cols = m.group(1).split('],')
        return len([c for c in cols if c.strip()])

    def extract_feature(self, node: dict) -> SampleEntity:
        # 如果需要 est 模式，请在此实现
        raise NotImplementedError("EST 模式尚未实现")

    def extract_pred_feature(self, node: dict) -> SampleEntity:
        op = node.get("OPERATOR", "").strip()
        width = self._parse_output_width(node.get("output", ""))

        # 先递归左右
        ch = self._get_children(node)
        left = self.extract_pred_feature(ch[0]) if len(ch) >= 1 else None
        right = self.extract_pred_feature(ch[1]) if len(ch) == 2 else None

        # 提取谓词（目前总是空）
        preds = extract_predicates_from_node(node)
        cols, ops = [], []
        for c, oper in preds:
            if c not in self.columns:
                self.columns.append(c)
            cols.append(self.columns.index(c))
            ops.append(OPERATORS.index(oper))
        enc = np.zeros(2 * self.max_pred_len + 1, dtype=int)
        enc[:len(cols)] = cols
        enc[self.max_pred_len:self.max_pred_len+len(ops)] = ops
        enc[-1] = len(cols)

        # 构造 SampleEntity
        return SampleEntity(
            node_type=op_to_one_hot(op),
            width=width,
            pred_encoding=enc,
            left=left,
            right=right
        )


class FeatureGenerator:
    """同时支持 est 模式和 pred 模式："""
    def __init__(self, use_est: bool = True, use_pred: bool = False):
        self.use_est = use_est
        self.use_pred = use_pred
        self.columns = COLUMNS.copy()
        self.input_relations = []
        self.feature_parser: OBAnalyzeJsonParser = None

    def fit_pred_model(self, trees: list):
        """扫描所有树，收集 predicate 中列名（目前无），并初始化 parser"""
        # 如果需要，这里可遍历 trees 收集新的 self.columns
        self.feature_parser = OBAnalyzeJsonParser(
            input_relations=self.input_relations,
            columns=self.columns
        )

    def transform(self, trees: list) -> list:
        """把 JSON plan 树转换成 SampleEntity 树，并补全一元节点"""
        out = []
        for tree in trees:
            root = tree.get("Plan", tree) if isinstance(tree, dict) else tree

            if self.use_est:
                se = self.feature_parser.extract_feature(root)
            else:
                se = self.feature_parser.extract_pred_feature(root)

            self._pad_unary(se)
            out.append(se)

        return out

    def _pad_unary(self, node: SampleEntity):
        if node.left is None and node.right is None:
            return
        if node.left and not node.right:
            node.right = self._make_dummy()
        if node.right and not node.left:
            node.left = self._make_dummy()
        if node.left:
            self._pad_unary(node.left)
        if node.right:
            self._pad_unary(node.right)

    def _make_dummy(self) -> SampleEntity:
        zero_enc = np.zeros(2 * self.feature_parser.max_pred_len + 1, dtype=int)
        return SampleEntity(
            node_type=op_to_one_hot(UNKNOWN_OP_TYPE),
            width=0,
            pred_encoding=zero_enc,
            left=None,
            right=None
        )

    def transform_z(self, Z, params, preprocessing_infos):
        return preprocess_z(Z, params, preprocessing_infos)
