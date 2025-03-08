from functools import partial
import pandas as pd
import plotly.express as px
import sklearn
import xgboost as xgb
from pandarallel import pandarallel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
pandarallel.initialize(progress_bar=True)
import json
import sys
from collections import Counter, defaultdict
from enum import Enum, auto
from pathlib import Path
import numpy as np
from scipy.stats import entropy
ZEND_AST_SPECIAL_SHIFT = 6
ZEND_AST_IS_LIST_SHIFT = 7
ZEND_AST_NUM_CHILDREN_SHIFT = 8
np.set_printoptions(threshold=np.inf)  # 设置显示所有元素


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
def str_entropy(s):
    """简单计算文本混淆度（熵）"""
    # 统计每个字符出现的次数
    counts = Counter(s)
    # 转换成概率分布
    probs = np.array(list(counts.values())) / len(s)
    # 计算熵
    ent = entropy(probs, base=2)
    return ent

def process_name(name):
    """处理节点名

    Example:
        "[ZVAL:\"c99shexit\"]" -> "ZVAL:c99shexit"
    """
    return name.replace("]", "").replace("[", "").replace('"', "")

# 风险函数
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
    # ...
]

# 可控输入
sources = [
    "_GET",
    "_POST",
    "_COOKIE",
    "_REQUEST",
    "_FILES",
    "_SERVER",
    "_ENV",
    "_SESSION",
    # ...
]

def parse(content):
    """解析JSON文件，提取相关有效信息"""
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
        nonlocal kind2cnt
        nonlocal lineno2nodecnt
        nonlocal sink2cnt
        nonlocal source2cnt
        nonlocal linenos
        nonlocal vals
        nonlocal num_val_cutoff
        nonlocal num_val_tohex
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

    return (
        kind2cnt,
        lineno2nodecnt,
        sink2cnt,
        source2cnt,
        linenos,
        vals,
        num_val_cutoff,
        num_val_tohex,
    )

def process(p: Path):
    """特征抽取"""
    with p.open("r") as f:
        content = f.read()
    (
        kind2cnt,
        lineno2nodecnt,
        sink2cnt,
        source2cnt,
        linenos,
        vals,
        num_val_cutoff,
        num_val_tohex,
    ) = parse(content)

    f_kind2cnt = {k.name: 0 for k in PHPNodeKind}
    f_kind2cnt.update(kind2cnt)

    f_sink2cnt = {s: 0 for s in sinks}
    f_sink2cnt.update(sink2cnt)
    f_source2cnt = {s: 0 for s in sources}
    f_source2cnt.update(source2cnt)

    vals_entropy = [str_entropy(val) for val in vals]
    nodecnts = list(lineno2nodecnt.values())

    all_features = {
        "num_line": len(linenos),
        "max_lineno": max(linenos, default=0),
        "num_val_cutoff": num_val_cutoff,
        "num_val_tohex": num_val_tohex,
        "max_node_of_line": max(nodecnts, default=0), # max(单行节点数)
        "std_node_of_line": np.std(nodecnts) if len(nodecnts) > 0 else 0,
        "var_node_of_line": np.var(nodecnts) if len(nodecnts) > 0 else 0,
        "entropy": str_entropy("".join(vals)) if len(vals) > 0 else 0,
        "max_val_entropy": max(vals_entropy, default=0),
        "min_val_entropy": min(vals_entropy, default=0),
        "std_val_entropy": np.std(vals_entropy) if len(vals_entropy) > 0 else 0,
        "var_val_entropy": np.var(vals_entropy) if len(vals_entropy) > 0 else 0,
        "vals": vals,
    }

    all_features.update(f_kind2cnt) # 节点统计
    all_features.update(f_sink2cnt) # 危险函数统计
    all_features.update(f_source2cnt) # 输入源统计

    return all_features

import zipfile

dataset_folder = Path("/Users/guyuwei/PycharmProjects/PythonProject/deep_learning/webshell检测/xgb")


train_label_file = dataset_folder / "train.csv"
train_file_folder = dataset_folder / "train"
test_file_folder = dataset_folder / "test"

def fid_transform(file_id, train_or_test="train"):
    p = (train_file_folder if train_or_test == "train" else test_file_folder) / str(
        file_id
    )
    return process(p)

# metric
fbeta_score = partial(sklearn.metrics.fbeta_score, beta=0.5, average="binary")

# label mapping
id2label = {0: "white", 1: "black"}
label2id = {v: k for k, v in id2label.items()}
label_encode = lambda x: label2id[x]
label_decode = lambda x: id2label[x]

# 自定义的评估函数
def precision_recall_fbeta(preds, dtrain, threshold=0.5):
    labels = dtrain.get_label()
    preds = preds > threshold  # 把预测概率转换成二分类结果
    acc = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    fbeta = fbeta_score(labels, preds)
    return [("precision", acc), ("recall", recall), ("fbeta", fbeta)]

# read label file
df = pd.read_csv(train_label_file)
df = df.loc[df["type"] == "php"]
df.sample(n=1000)

y = df["label"].map(label_encode)
_X = df["file_id"].parallel_map(fid_transform)

X = pd.DataFrame(_X.copy().tolist())
X = X.drop("vals", axis=1)

# 检查结果
X[X.isna().any(axis=1)]

# 拆分训练集&验证集
Xtrain, Xeval, ytrain, yeval = train_test_split(
    X, y, test_size=0.1, random_state=1337
)
dtrain = xgb.DMatrix(data=Xtrain.values, feature_names=list(Xtrain.columns), label=ytrain)
deval = xgb.DMatrix(data=Xeval, label=yeval)
print(f"{deval.num_row()=}")

watchlist = [(deval, "eval"), (dtrain, "train")]


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

xgm = xgb.train(param, dtrain, num_round, evals=watchlist, feval=precision_recall_fbeta)
xgm.save_model('xgb_model.json')

fi = dict(
    sorted(xgm.get_score(importance_type="total_gain").items(), key=lambda x: x[1])
)
px.bar(y=list(fi.keys())[-20:], x=list(fi.values())[-20:])

e=xgm.predict(deval)
e_binary = (e > 0.5).astype(int)  # 将概率值转换为二分类标签
print("二分类结果：")
print(e_binary)

print("预测出的概率值：")
print(e)

fig = px.histogram(x=xgm.predict(deval), labels={'x':'probs', 'y':'count'})
fig.show()