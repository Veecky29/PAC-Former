import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import csv
import gc
import hashlib
import itertools
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from .model import create_model as build_model
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data/benchmark'
DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Traffic']
HORIZONS = [96, 192, 336, 720]
GRID = [dict(learning_rate=lr, d_model=dm, kernel_size=k) for lr, dm, k in itertools.product([0.0001, 0.0005, 0.001], [128, 256], [5, 7])]
ACTIVE_VARIANT = 'F'

def create_model(channels, horizon, config):
    return build_model(channels, horizon, config, variant=ACTIVE_VARIANT)

def utc():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    temp.replace(path)

def atomic_model(path, data):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(data, temp)
    temp.replace(path)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; CPU fallback is disabled')
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

def lr_at(step, steps, peak):
    if step < steps:
        return peak * (0.1 + 0.9 * step / max(1, steps - 1))
    progress = (step - steps) / max(1, 50 * steps - steps - 1)
    return peak * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))

class Series:

    def __init__(self, name, audit, run_dir):
        self.name = name
        self.audit = audit['datasets'][name]
        p = DATA_DIR / (name + '.csv')
        assert sha(p) == self.audit['sha256']
        frame = pd.read_csv(p)
        self.columns = self.audit['columns']
        self.raw = frame[self.columns].to_numpy(dtype=np.float64)
        s = self.audit['splits_steps']
        self.ntrain = s['train']
        self.nval = s['val']
        self.ntest = s['test']
        self.used = self.ntrain + self.nval + self.ntest
        self.channels = self.raw.shape[1]
        self.scaler = StandardScaler().fit(self.raw[:self.ntrain])
        self.scaled = self.scaler.transform(self.raw[:self.used]).astype(np.float32)
        assert np.isfinite(self.scaled).all() and int(self.scaler.n_samples_seen_) == self.ntrain
        np.testing.assert_allclose(self.scaler.mean_, self.raw[:self.ntrain].mean(axis=0), rtol=1e-12, atol=1e-12)
        self.gpu = torch.from_numpy(self.scaled).to('cuda')
        self.xoffset = torch.arange(96, device='cuda')
        self.yoffset = {h: torch.arange(h, device='cuda') + 96 for h in HORIZONS}
        sp = run_dir / 'data_protocol' / name
        sp.mkdir(parents=True, exist_ok=True)
        np.savez(sp / 'scaler.npz', mean=self.scaler.mean_, scale=self.scaler.scale_, var=self.scaler.var_, n_samples_seen=self.scaler.n_samples_seen_)

    def bounds(self, h, split):
        if split == 'train':
            start, end = (0, self.ntrain)
        elif split == 'val':
            start, end = (self.ntrain - 96, self.ntrain + self.nval)
        elif split == 'test':
            start, end = (self.ntrain + self.nval - 96, self.used)
        else:
            raise ValueError(split)
        return (start, end - 96 - h + 1)

    def batch(self, starts, h):
        idx = starts.to('cuda')
        x = self.gpu[idx[:, None] + self.xoffset[None, :]]
        y = self.gpu[idx[:, None] + self.yoffset[h][None, :]]
        return (x, y)

    def close(self):
        del self.gpu
        self.yoffset.clear()
        del self.xoffset
        gc.collect()
        torch.cuda.empty_cache()

class FullErrors:

    def __init__(self):
        self.sse = torch.zeros((), dtype=torch.float64, device='cuda')
        self.sae = torch.zeros((), dtype=torch.float64, device='cuda')
        self.elements = 0
        self.windows = 0

    def add(self, pred, true):
        assert pred.shape == true.shape
        error = pred.double() - true.double()
        self.sse += error.square().sum()
        self.sae += error.abs().sum()
        self.elements += true.numel()
        self.windows += true.shape[0]

    def result(self):
        if self.elements <= 0:
            raise RuntimeError('Empty evaluation')
        mse = float((self.sse / self.elements).item())
        mae = float((self.sae / self.elements).item())
        if not np.isfinite([mse, mae]).all():
            raise FloatingPointError('Non-finite full-sample metric')
        return dict(mse=mse, mae=mae, windows=self.windows, elements=self.elements)

def evaluate(model, series, h, split, batch_size=32):
    model.eval()
    a, b = series.bounds(h, split)
    acc = FullErrors()
    with torch.no_grad():
        for start in range(a, b, batch_size):
            idx = torch.arange(start, min(b, start + batch_size), dtype=torch.long)
            x, y = series.batch(idx, h)
            pred = model(x)
            acc.add(pred, y)
    out = acc.result()
    assert out['windows'] == b - a
    assert out['elements'] == (b - a) * h * series.channels
    out['last_batch_windows'] = (b - a) % batch_size or batch_size
    return out

def fit_job(series, h, cfg, seed, path, kind, protocol_hash, log, status_cb):
    result_path = path / 'result.json'
    path.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding='utf8'))
        assert result['protocol_hash'] == protocol_hash and result['config'] == cfg and (result['seed'] == seed)
        if result['status'] == 'completed':
            assert (path / 'best.pt').exists()
            log.info('RESUME skip completed %s', path.name)
            return result
    seed_all(seed)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    model = create_model(series.channels, h, cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    a, b = series.bounds(h, 'train')
    n = b - a
    steps = n // 32
    assert steps > 0
    best = float('inf')
    best_epoch = 0
    patience = 0
    epochs = []
    started = time.perf_counter()
    checkpoint = path / 'best.pt'
    torch.cuda.reset_peak_memory_stats()
    start_epoch = 1
    if (path / 'last.pt').exists():
        saved = torch.load(path / 'last.pt', map_location='cpu', weights_only=False)
        assert saved['protocol_hash'] == protocol_hash and saved['config'] == cfg and (saved['seed'] == seed)
        model.load_state_dict(saved['model_state'])
        optimizer.load_state_dict(saved['optimizer_state'])
        generator.set_state(saved['shuffle_rng'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng'])
        best = saved['best']
        best_epoch = saved['best_epoch']
        patience = saved['patience']
        epochs = saved['epochs']
        start_epoch = saved['epoch'] + 1
        del saved
    if patience >= 5:
        start_epoch = 51
    job_meta = dict(dataset=series.name, horizon=h, config=cfg, seed=seed, kind=kind, protocol_hash=protocol_hash)
    with (path / 'epochs.jsonl').open('w', encoding='utf8', buffering=1) as epoch_log:
        for previous in epochs:
            epoch_log.write(json.dumps(previous, allow_nan=False) + '\n')
        for epoch in range(start_epoch, 51):
            epoch_start = time.perf_counter()
            model.train()
            order = torch.randperm(n, generator=generator) + a
            train_sum = torch.zeros((), dtype=torch.float64, device='cuda')
            for i in range(steps):
                lr = lr_at((epoch - 1) * steps + i, steps, cfg['learning_rate'])
                for g in optimizer.param_groups:
                    g['lr'] = lr
                x, y = series.batch(order[i * 32:(i + 1) * 32], h)
                optimizer.zero_grad()
                prediction = model(x)
                loss = torch.nn.functional.mse_loss(prediction, y)
                loss.backward()
                optimizer.step()
                train_sum += loss.detach().double()
            train_mse = float((train_sum / steps).item())
            if not np.isfinite(train_mse):
                raise FloatingPointError(f'Non-finite training loss: {job_meta}')
            val = evaluate(model, series, h, 'val')
            improved = val['mse'] < best
            if improved:
                best = val['mse']
                best_epoch = epoch
                patience = 0
                payload = dict(model_state={k: v.detach().cpu() for k, v in model.state_dict().items()}, model_config=dict(c_in=series.channels, seq_len=96, pred_len=h, d_model=cfg['d_model'], n_heads=8, e_layers=2, d_ff=2 * cfg['d_model'], dropout=0.1, kernel_size=cfg['kernel_size'], d_scale=16), selection=dict(**job_meta, best_epoch=best_epoch, best_val_mse=best, data_sha256=series.audit['sha256'], channels=series.columns, scaler_path=str(Path('data_protocol') / series.name / 'scaler.npz')))
                atomic_model(checkpoint, payload)
            else:
                patience += 1
            record = dict(epoch=epoch, lr_first=lr_at((epoch - 1) * steps, steps, cfg['learning_rate']), lr_last=lr, train_mse=train_mse, train_windows_used=steps * 32, train_windows_dropped=n - steps * 32, val=val, improved=improved, seconds=time.perf_counter() - epoch_start, utc=utc())
            epochs.append(record)
            epoch_log.write(json.dumps(record, allow_nan=False) + '\n')
            atomic_model(path / 'last.pt', dict(protocol_hash=protocol_hash, config=cfg, seed=seed, epoch=epoch, model_state={k: v.detach().cpu() for k, v in model.state_dict().items()}, optimizer_state=optimizer.state_dict(), shuffle_rng=generator.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=np.random.get_state(), python_rng=random.getstate(), best=best, best_epoch=best_epoch, patience=patience, epochs=epochs))
            status_cb(dict(**job_meta, epoch=epoch, val_mse=val['mse'], best_val_mse=best))
            log.info('%s H%d %s seed%d epoch %d valMSE %.9f best %.9f %.2fs', series.name, h, path.name, seed, epoch, val['mse'], best, record['seconds'])
            if patience >= 5:
                break
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model_state'], strict=True)
    selected_val = evaluate(model, series, h, 'val')
    assert selected_val['mse'] == best
    test = evaluate(model, series, h, 'test') if kind == 'final_refit' else None
    if kind == 'final_refit':
        for split, reference in [('val', selected_val), ('test', test)]:
            check = evaluate(model, series, h, split, 17)
            assert check['elements'] == reference['elements'] and check['windows'] == reference['windows']
            np.testing.assert_allclose([check['mse'], check['mae']], [reference['mse'], reference['mae']], rtol=2e-06, atol=2e-07)
    result = dict(status='completed', **job_meta, best_val_mse=best, best_epoch=best_epoch, epochs_ran=len(epochs), train_windows_available=n, train_windows_used_per_epoch=steps * 32, val=selected_val, test=test, parameters=sum((p.numel() for p in model.parameters())), peak_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20, peak_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20, elapsed_seconds=sum((e['seconds'] for e in epochs)), checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), ended_utc=utc())
    atomic_json(result_path, result)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result

def write_csv(path, rows):
    if not rows:
        return
    temp = path.with_suffix('.tmp')
    with temp.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    temp.replace(path)

def reuse_main_checkpoint(source, series, horizon, config, path, protocol_hash):
    payload = torch.load(source, map_location='cpu', weights_only=True)
    selection = payload['selection']
    assert selection['dataset'] == series.name and selection['horizon'] == horizon
    assert selection['seed'] == 2026 and selection['config'] == config
    assert selection['data_sha256'] == series.audit['sha256']
    model = build_model(series.channels, horizon, config)
    model.load_state_dict(payload['model_state'], strict=True)
    for split in ['val', 'test']:
        measured = evaluate(model, series, horizon, split)
        expected = payload['metrics'][split]
        np.testing.assert_allclose([measured['mse'], measured['mae']], [expected['mse'], expected['mae']], rtol=2e-06, atol=2e-07)
    path.mkdir(parents=True, exist_ok=True)
    atomic_model(path / 'best.pt', payload)
    result = dict(status='completed', dataset=series.name, horizon=horizon, config=config, seed=2026, kind='final_refit', protocol_hash=protocol_hash, best_epoch=selection['best_epoch'], best_val_mse=selection['best_val_mse'], val=payload['metrics']['val'], test=payload['metrics']['test'], **payload['training'], checkpoint=str(path / 'best.pt'), checkpoint_sha256=sha(path / 'best.pt'))
    atomic_json(path / 'result.json', result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result
