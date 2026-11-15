import torch
import torch.nn as nn

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_config, get_weights_file_path, latest_weights_file_path

from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):

    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_ds(config):
    # The pair is only published as 'ar-en'; lang_src/lang_tgt pick the translation direction
    ds_raw = load_dataset('Helsinki-NLP/news_commentary', 'ar-en', split='train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Drop pairs that don't fit in seq_len: +2 for [SOS]/[EOS] on the encoder side, +1 on the decoder side
    def fits_seq_len(batch):
        src_encodings = tokenizer_src.encode_batch([pair[config['lang_src']] for pair in batch['translation']])
        tgt_encodings = tokenizer_tgt.encode_batch([pair[config['lang_tgt']] for pair in batch['translation']])
        return [len(src.ids) + 2 <= config['seq_len'] and len(tgt.ids) + 1 <= config['seq_len']
                for src, tgt in zip(src_encodings, tgt_encodings)]

    num_before = len(ds_raw)
    ds_raw = ds_raw.filter(fits_seq_len, batched=True)
    print(f'Kept {len(ds_raw)} of {num_before} pairs that fit in seq_len={config["seq_len"]}')

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    # Fixed seed: a resumed run must get the same split, or validation pairs leak into training
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size,val_ds_size], generator=torch.Generator().manual_seed(42))

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw: 
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src,len(src_ids))
        max_len_tgt = max(max_len_tgt,len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size = config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'],config['d_model'])
    return model


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # The encoder runs once; the decoder reruns on the growing output, one new token per step
    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.full((1, 1), sos_idx, dtype=source.dtype, device=device)

    while decoder_input.size(1) < max_len:
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        next_word = model.project(out[:, -1]).argmax(dim=-1, keepdim=True)
        decoder_input = torch.cat([decoder_input, next_word], dim=1)
        if next_word.item() == eos_idx:
            break

    return decoder_input.squeeze(0)


def compute_loss(model, batch, loss_fn, tgt_vocab_size, device):
    encoder_input = batch['encoder_input'].to(device)  # (B, seq_len)
    decoder_input = batch['decoder_input'].to(device)  # (B, seq_len)
    encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, seq_len)
    decoder_mask = batch['decoder_mask'].to(device)  # (B, 1, seq_len, seq_len)
    label = batch['label'].to(device)  # (B, seq_len)

    encoder_output = model.encode(encoder_input, encoder_mask)
    decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
    proj_output = model.project(decoder_output)  # (B, seq_len, tgt_vocab_size)
    return loss_fn(proj_output.view(-1, tgt_vocab_size), label.view(-1))


def run_validation(model, val_dataloader, tokenizer_tgt, max_len, device, writer, global_step, num_examples=2):
    model.eval()
    examples = []
    with torch.no_grad():
        for count, batch in enumerate(val_dataloader, start=1):
            encoder_input = batch['encoder_input'].to(device)  # (1, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)  # (1, 1, 1, seq_len)
            assert encoder_input.size(0) == 1, 'Validation batch size must be 1'

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device)
            predicted = tokenizer_tgt.decode(model_out.tolist())

            print('-' * 80)
            print(f'SOURCE:    {batch["src_text"][0]}')
            print(f'TARGET:    {batch["tgt_text"][0]}')
            print(f'PREDICTED: {predicted}')
            examples.append(f'**SOURCE:** {batch["src_text"][0]}  \n**TARGET:** {batch["tgt_text"][0]}  \n**PREDICTED:** {predicted}')

            if count == num_examples:
                break

    writer.add_text('validation/examples', '\n\n---\n\n'.join(examples), global_step)


def run_validation_loss(model, val_loss_dataloader, loss_fn, tgt_vocab_size, device, use_amp):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loss_dataloader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                total_loss += compute_loss(model, batch, loss_fn, tgt_vocab_size, device).item()
    return total_loss / len(val_loss_dataloader)


def train_model(config):
    device = get_device()
    print(f'Using device: {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    tgt_vocab_size = tokenizer_tgt.get_vocab_size()
    model = get_model(config, tokenizer_src.get_vocab_size(), tgt_vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # Mixed precision only on CUDA (Colab): roughly 2x faster on T4/A100 tensor cores
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    if config['preload'] == 'latest':
        model_filename = latest_weights_file_path(config)
    elif config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
    else:
        model_filename = None

    initial_epoch = 0
    global_step = 0
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        if 'scaler_state_dict' in state:
            scaler.load_state_dict(state['scaler_state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
    else:
        print('No checkpoint to preload, starting from scratch')

    writer = SummaryWriter(config['experiment_name'])
    val_loss_dataloader = DataLoader(val_dataloader.dataset, batch_size=config['batch_size'])

    # ProjectionLayer already returns log-probs. CrossEntropyLoss's own log_softmax leaves log-probs
    # unchanged, so the loss is the same as NLLLoss, and unlike NLLLoss it supports label smoothing.
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'), label_smoothing=0.1)

    for epoch in range(initial_epoch, config['num_epoch']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Epoch {epoch:02d}')
        for batch in batch_iterator:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = compute_loss(model, batch, loss_fn, tgt_vocab_size, device)
            batch_iterator.set_postfix(loss=f'{loss.item():6.3f}')
            writer.add_scalar('loss/train', loss.item(), global_step)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

        val_loss = run_validation_loss(model, val_loss_dataloader, loss_fn, tgt_vocab_size, device, use_amp)
        print(f'Epoch {epoch:02d} validation loss: {val_loss:.3f}')
        writer.add_scalar('loss/validation', val_loss, global_step)
        run_validation(model, val_dataloader, tokenizer_tgt, config['seq_len'], device, writer, global_step)
        writer.flush()

        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }, get_weights_file_path(config, f'{epoch:02d}'))

    writer.close()


if __name__ == '__main__':
    train_model(get_config())