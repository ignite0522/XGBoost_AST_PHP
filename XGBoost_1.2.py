import zipfile
import pandas as pd
import plotly.express as px
import sklearn
import xgboost as xgb
from pandarallel import pandarallel
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
import warnings
import json
import sys
from collections import Counter, defaultdict
from enum import Enum, auto
from pathlib import Path
import numpy as np
from scipy.stats import entropy
from functools import partial

warnings.filterwarnings('ignore')
pandarallel.initialize(progress_bar=True)

# 定义PHP AST节点类型
ZEND_AST_SPECIAL_SHIFT = 6
ZEND_AST_IS_LIST_SHIFT = 7
ZEND_AST_NUM_CHILDREN_SHIFT = 8

class PHPNodeKind(Enum):
    # special nodes
    ZEND_AST_ZVAL = 1 << ZEND_AST_SPECIAL_SHIFT
    ZEND_AST_CONSTANT = auto()
    ZEND_AST_ZNODE = auto()

    # declaration nodes
    ZEND_AST_FUNC_DECL = auto()
    ZEND_AST_CLOSURE = auto()
    ZEND_AST_METHOD = auto()
    ZEND_AST_CLASS = auto()
    ZEND_AST_ARROW_FUNC = auto()

    # list nodes
    ZEND_AST_ARG_LIST = 1 << ZEND_AST_IS_LIST_SHIFT
    ZEND_AST_ARRAY = auto()
    ZEND_AST_ENCAPS_LIST = auto()
    ZEND_AST_EXPR_LIST = auto()
    ZEND_AST_STMT_LIST = auto()
    ZEND_AST_IF = auto()
    ZEND_AST_SWITCH_LIST = auto()
    ZEND_AST_CATCH_LIST = auto()
    ZEND_AST_PARAM_LIST = auto()
    ZEND_AST_CLOSURE_USES = auto()
    ZEND_AST_PROP_DECL = auto()
    ZEND_AST_CONST_DECL = auto()
    ZEND_AST_CLASS_CONST_DECL = auto()
    ZEND_AST_NAME_LIST = auto()
    ZEND_AST_TRAIT_ADAPTATIONS = auto()
    ZEND_AST_USE = auto()

    # 0 child nodes
    ZEND_AST_MAGIC_CONST = 0 << ZEND_AST_NUM_CHILDREN_SHIFT
    ZEND_AST_TYPE = auto()
    ZEND_AST_CONSTANT_CLASS = auto()

    # 1 child node
    ZEND_AST_VAR = 1 << ZEND_AST_NUM_CHILDREN_SHIFT
    ZEND_AST_CONST = auto()
    ZEND_AST_UNPACK = auto()
    ZEND_AST_UNARY_PLUS = auto()
    ZEND_AST_UNARY_MINUS = auto()
    ZEND_AST_CAST = auto()
    ZEND_AST_EMPTY = auto()
    ZEND_AST_ISSET = auto()
    ZEND_AST_SILENCE = auto()
    ZEND_AST_SHELL_EXEC = auto()
    ZEND_AST_CLONE = auto()
    ZEND_AST_EXIT = auto()
    ZEND_AST_PRINT = auto()
    ZEND_AST_INCLUDE_OR_EVAL = auto()
    ZEND_AST_UNARY_OP = auto()
    ZEND_AST_PRE_INC = auto()
    ZEND_AST_PRE_DEC = auto()
    ZEND_AST_POST_INC = auto()
    ZEND_AST_POST_DEC = auto()
    ZEND_AST_YIELD_FROM = auto()
    ZEND_AST_CLASS_NAME = auto()

    ZEND_AST_GLOBAL = auto()
    ZEND_AST_UNSET = auto()
    ZEND_AST_RETURN = auto()
    ZEND_AST_LABEL = auto()
    ZEND_AST_REF = auto()
    ZEND_AST_HALT_COMPILER = auto()
    ZEND_AST_ECHO = auto()
    ZEND_AST_THROW = auto()
    ZEND_AST_GOTO = auto()
    ZEND_AST_BREAK = auto()
    ZEND_AST_CONTINUE = auto()

    # 2 child nodes
    ZEND_AST_DIM = 2 << ZEND_AST_NUM_CHILDREN_SHIFT
    ZEND_AST_PROP = auto()
    ZEND_AST_STATIC_PROP = auto()
    ZEND_AST_CALL = auto()
    ZEND_AST_CLASS_CONST = auto()
    ZEND_AST_ASSIGN = auto()
    ZEND_AST_ASSIGN_REF = auto()
    ZEND_AST_ASSIGN_OP = auto()
    ZEND_AST_BINARY_OP = auto()
    ZEND_AST_GREATER = auto()
    ZEND_AST_GREATER_EQUAL = auto()
    ZEND_AST_AND = auto()
    ZEND_AST_OR = auto()
    ZEND_AST_ARRAY_ELEM = auto()
    ZEND_AST_NEW = auto()
    ZEND_AST_INSTANCEOF = auto()
    ZEND_AST_YIELD = auto()
    ZEND_AST_COALESCE = auto()
    ZEND_AST_ASSIGN_COALESCE = auto()

    ZEND_AST_STATIC = auto()
    ZEND_AST_WHILE = auto()
    ZEND_AST_DO_WHILE = auto()
    ZEND_AST_IF_ELEM = auto()
    ZEND_AST_SWITCH = auto()
    ZEND_AST_SWITCH_CASE = auto()
    ZEND_AST_DECLARE = auto()
    ZEND_AST_USE_TRAIT = auto()
    ZEND_AST_TRAIT_PRECEDENCE = auto()
    ZEND_AST_METHOD_REFERENCE = auto()
    ZEND_AST_NAMESPACE = auto()
    ZEND_AST_USE_ELEM = auto()
    ZEND_AST_TRAIT_ALIAS = auto()
    ZEND_AST_GROUP_USE = auto()
    ZEND_AST_PROP_GROUP = auto()

    # 3 child nodes
    ZEND_AST_METHOD_CALL = 3 << ZEND_AST_NUM_CHILDREN_SHIFT
    ZEND_AST_STATIC_CALL = auto()
    ZEND_AST_CONDITIONAL = auto()

    ZEND_AST_TRY = auto()
    ZEND_AST_CATCH = auto()
    ZEND_AST_PARAM = auto()
    ZEND_AST_PROP_ELEM = auto()
    ZEND_AST_CONST_ELEM = auto()

    # 4 child nodes
    ZEND_AST_FOR = 4 << ZEND_AST_NUM_CHILDREN_SHIFT
    ZEND_AST_FOREACH = auto()

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

    @classmethod
    def decode(cls, id):
        return cls(id).name

# 特征提取函数：计算字符串熵
def str_entropy(s):
    counts = Counter(s)
    probs = np.array(list(counts.values())) / len(s)
    ent = entropy(probs, base=2)
    return ent

def process_name(name):
    """处理节点名"""
    return name.replace("]", "").replace("[", "").replace('"', "")

# 风险函数（危险代码）
sinks = [
    "exec",
    "passthru",
    "shell_exec",
    "system",
    "ssh2_exec",
    "INCLUDE_OR_EVAL(eval)",
    "INCLUDE_OR_EVAL(include)",
    "INCLUDE_OR_EVAL(include_once)",
    "INCLUDE_OR_EVAL(require)",
    "INCLUDE_OR_EVAL(require_once)",
    "file_get_contents",
    "assert",
    "create_function",
]

# 可控输入源
sources = [
    "_GET",
    "_POST",
    "_COOKIE",
    "_REQUEST",
    "_FILES",
    "_SERVER",
    "_ENV",
    "_SESSION",
]

# 解析JSON文件并提取相关信息
def parse(content):
    try:
        tree = json.loads(content)
    except Exception as err:
        print(f"Warning: {err}, {type(err)}")
        sys.exit(1)

    kind2cnt = defaultdict(int)
    lineno2nodecnt = defaultdict(int)
    sink2cnt = defaultdict(int)
    source2cnt = defaultdict(int)
    linenos = set()
    vals = []
    num_val_cutoff = 0
    num_val_tohex = 0

    def dfs(node):
        nonlocal kind2cnt, lineno2nodecnt, sink2cnt, source2cnt, linenos, vals, num_val_cutoff, num_val_tohex
        if "kind" in node and PHPNodeKind.has_value(node["kind"]):
            kind2cnt[PHPNodeKind.decode(node["kind"])] += 1
        if "lineno" in node:
            v = node["lineno"]
            linenos.add(v)
            lineno2nodecnt[v] += 1
        if "val" in node and isinstance(node["val"], str):
            v = node["val"]
            vals.append(v)
            for s in sources:
                source2cnt[s] += int(s == v)
        if "name" in node:
            v = process_name(node["name"])
            if ":" in v:
                sep_idx = v.index(":")
                vk, vv = v[0:sep_idx], v[sep_idx + 1 :]
                for s in sinks:
                    sink2cnt[s] += int(s == vk) + int(s == vv)
        if "val_cutoff" in node:
            num_val_cutoff += int(node["val_cutoff"])
        if "val_tohex" in node:
            num_val_tohex += int(node["val_tohex"])
        if "children" in node:
            [dfs(child) for child in node["children"]]

    dfs(tree)

    return kind2cnt, lineno2nodecnt, sink2cnt, source2cnt, linenos, vals, num_val_cutoff, num_val_tohex

def process(p: Path):
    with p.open("r") as f:
        content = f.read()
    kind2cnt, lineno2nodecnt, sink2cnt, source2cnt, linenos, vals, num_val_cutoff, num_val_tohex = parse(content)

    f_kind2cnt = {k.name: 0 for k in PHPNodeKind}
    f_kind2cnt.update(kind2cnt)

    f_sink2cnt = {s: 0 for s in sinks}
    f_sink2cnt.update(sink2cnt)

    f_source2cnt = {s: 0 for s in sources}
    f_source2cnt.update(source2cnt)

    return {
        "kind2cnt": f_kind2cnt,
        "sink2cnt": f_sink2cnt,
        "source2cnt": f_source2cnt,
        "num_val_cutoff": num_val_cutoff,
        "num_val_tohex": num_val_tohex,
        "linenos": len(linenos),
        "vals": vals,
        "entropy": str_entropy("".join(vals)),
    }

# 数据路径设置
dataset_folder = Path("/PythonProject/deep_learning/webshell检测/xgb")

train_label_file = dataset_folder / "train.csv"
train_file_folder = dataset_folder / "train"
test_file_folder = dataset_folder / "test"

def fid_transform(file_id, train_or_test="train"):
    """特征提取函数"""
    p = (train_file_folder if train_or_test == "train" else test_file_folder) / str(file_id)
    return process(p)

def precision_recall_fbeta(preds, dtrain, threshold=0.5):
    labels = dtrain.get_label()
    preds = preds > threshold  # 把预测概率转换成二分类结果
    acc = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    fbeta = fbeta_score(labels, preds)
    return [("precision", acc), ("recall", recall), ("fbeta", fbeta)]

# metric
fbeta_score = partial(sklearn.metrics.fbeta_score, beta=0.5, average="binary")

# label mapping
id2label = {0: "white", 1: "black"}
label2id = {v: k for k, v in id2label.items()}
label_encode = lambda x: label2id[x]
label_decode = lambda x: id2label[x]

# 读取标签文件
df = pd.read_csv(train_label_file)
df = df.loc[df["type"] == "php"]  # 只选择php类型的
df.sample(n=1000)

# 标签编码
y = df["label"].map(label_encode)

# 特征提取
_X = df["file_id"].parallel_map(fid_transform)
X = pd.DataFrame(_X.copy().tolist())
X = X.drop("vals", axis=1)

# 创建训练集
dtrain = xgb.DMatrix(data=X.values, feature_names=list(X.columns), label=y)

# 定义训练参数
param = {
    "max_depth": 20,
    "tree_method": "hist",
    "eta": 1,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "verbose": True,
    "early_stopping_rounds": 10,
}
num_round = 10

# 训练模型
xgm = xgb.train(param, dtrain, num_round, feval=precision_recall_fbeta)

# 保存模型
xgm.save_model('xgb_model.json')

# 特征重要性可视化
fi = dict(
    sorted(xgm.get_score(importance_type="total_gain").items(), key=lambda x: x[1])
)
px.bar(y=list(fi.keys())[-20:], x=list(fi.values())[-20:])

# 测试：预测未知文件（例如从测试集）
test_file_id = '/PythonProject/deep_learning/webshell检测/xgb/train/2'  # 设置你要测试的未知文件ID
test_features = fid_transform(test_file_id, train_or_test="test")  # 提取未知文件特征

# 将提取的特征转换为 DMatrix
dtest = xgb.DMatrix(data=[test_features], feature_names=list(X.columns))

# 使用训练好的模型进行预测
preds = xgm.predict(dtest)

# 输出预测结果
print(f"Prediction for test file {test_file_id}: {preds}")
