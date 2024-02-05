from numpy.typing import ArrayLike
from numpy import array
from pathlib import Path
from json import JSONDecodeError, load as j_load


def read_val_loss_from_log(path_to_log: Path) -> ArrayLike:
    results = []
    with open(path_to_log, "r") as fin:
        for line in fin:
            if "val loss" not in line:
                continue
            parts = line.strip().split()
            results.append(
                (int(parts[1][:-1]), float(parts[-1]))
            )
    return array(results)


def read_kaggle_log(fp):
    try:
        with open(fp, "r") as fin:
            return j_load(fin)
    except JSONDecodeError:
        nblog = []
        with open(fp, "r") as fin:
            for line in fin:
                parts = line.strip().split()
                try:
                    nblog.append(dict(
                        time=float(parts[0].rstrip("s")),
                        line_number=int(parts[1]),
                        data=" ".join(parts[2:])
                    ))
                except Exception as e:
                    continue
    return nblog


def parse_kaggle_log(fp):
    nblog = read_kaggle_log(fp)
    model, vloss = {}, []
    for logline in nblog:
        ldata = logline["data"].strip()
        if "val loss" in ldata:
            parts = ldata.split()
            vloss.append(
                (int(parts[1][:-1]), float(parts[-1]))
            )
        if "Overriding: n_" in ldata:
            parts = ldata.split()
            mp = parts[1]
            if mp not in model:
                model[mp] = int(parts[-1])
        if "number of parameters" in ldata:
            if "params" not in model:
                model["params"] = ldata.split()[-1]
    return model, array(vloss)
