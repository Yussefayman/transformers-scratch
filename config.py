from pathlib import Path

def get_config():
    return {
        "batch_size":8,
        "num_epoch": 50,
        "lr": 10**-4,
        "seq_len": 128,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "ar",
        "model_folder":"weights",
        "model_basename":"tmodel_",
        "preload":None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name":"runs/tmodel"
    }

def get_weights_file_path(config, epoch:str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f'{model_basename}{epoch}.pt'
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
    # Epochs are saved zero-padded (00, 01, ...), so name order is epoch order
    weights_files = sorted(Path(config['model_folder']).glob(f"{config['model_basename']}*.pt"))
    return str(weights_files[-1]) if weights_files else None
    