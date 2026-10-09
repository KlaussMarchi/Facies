import papermill as pm
from pathlib import Path
import os, json, csv, sys

sys.path.append('..')
from Dataset.index import Normalization

 
def execute(path):
    p = Path(path)
    dir_path, name, ext = p.parent, p.stem, p.suffix
    print('etapa: ', name)
    
    os.makedirs('logs', exist_ok=True)
    out = os.path.join('logs', f'{name}_out{ext}')    
    
    try:
        pm.execute_notebook(path, out, kernel_name='python3', log_output=True, progress_bar=True, cwd=str(dir_path))
        return True
    except Exception as e:
        print(f'Error executing {path}: {e}')
        return False


# OS TILES DO DataBase.csv SERVEM QUANDO FORAM CORTADOS COM O img_size E O step DA RODADA (null ACEITA O QUE EXISTIR)
# E ESTÃO NO ESCALONAMENTO QUE ELA PEDE (SEM A COLUNA scaling, É O 'percentile', O VOLUME COMO VEM)
def formatted(database, info):
    if not os.path.exists(database):
        return False

    with open(database, 'r') as file:
        row = next(csv.DictReader(file))

    shape = str(tuple(info['img_size'])) if info.get('img_size') else row.get('shape')
    step  = str(info['step']) if info.get('step') else row.get('step')
    return (row.get('shape') == shape) and (row.get('step') == step) and (Normalization.read(database) == info.get('scaling', Normalization.DEFAULT))


with open('task.json', 'r') as file:
    tasks = json.load(file)

for i, task in enumerate(tasks):
    print(f'\n\nRodada {i+1}/{len(tasks)}')

    with open('info.json', 'w') as file:
        file.write(json.dumps(task))

    with open('info.json', 'r') as file:
        info = json.load(file)
    
    print('info: ', info)
    dataset  = info.get('dataset')
    database = f"../Dataset/{dataset}/DataBase.csv"
    print(dataset)

    if formatted(database, info):
        print(f'Format pulado: {database} ja existe com img_size {info.get("img_size")}, step {info.get("step")} e scaling {info.get("scaling", Normalization.DEFAULT)!r}')
    else:
        execute(f"../Dataset/{dataset}/Format.ipynb")

    n_trials = int(info.get('n_trials') or 1)
    
    for trial in range(n_trials):
        print(f'\nTrial {trial+1}/{n_trials}')

        with open('info.json', 'w') as file:
            file.write(json.dumps({**task, 'trial': trial}))

        if execute("../Model/1 - Model.ipynb"):
            execute("../Model/3 - Predict.ipynb")

