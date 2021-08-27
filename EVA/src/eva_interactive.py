# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Redis Backend Generate EVA"""

USE_TORCH_DDP = False

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import time
from arguments import get_args
from utils import Timers
from utils import load_checkpoint
from tokenization_enc_dec import EncDecTokenizer
import mpu
import deepspeed
import torch.distributed as dist
from model import EncDecModel, EncDecConfig
from fp16 import FP16_Module
from utils import print_rank_0

if USE_TORCH_DDP:
    from torch.nn.parallel.distributed import DistributedDataParallel as DDP
else:
    from model import DistributedDataParallel as DDP


class EncDecModelForInference(EncDecModel):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
    def forward(
        self,
        enc_input_ids=None,
        enc_position_ids=None,
        enc_attention_mask=None,
        dec_input_ids=None,
        dec_position_ids=None,
        dec_attention_mask=None,
        cross_attention_mask=None,
        enc_hidden_states=None,
        past_key_values=None,
    ):
        if enc_hidden_states is None:
            enc_outputs = self.encoder(
                input_ids=enc_input_ids,
                position_ids=enc_position_ids,
                attention_mask=enc_attention_mask,
            )
            return enc_outputs
        
        else:
            dec_outputs = self.decoder(
                input_ids=dec_input_ids,
                position_ids=dec_position_ids,
                attention_mask=dec_attention_mask,
                cross_attention_mask=cross_attention_mask,
                enc_hidden_states=enc_hidden_states,
                past_key_values=past_key_values,
            )
            last_hidden_state_parallel = mpu.copy_to_model_parallel_region(dec_outputs["last_hidden_state"])
            logits_parallel = F.linear(last_hidden_state_parallel, self.lm_head.weight)
    
            if self.parallel_output:
                lm_logits = logits_parallel
            else:
                lm_logits = mpu.gather_from_model_parallel_region(logits_parallel)
                
            return dec_outputs, lm_logits


def calc_banned_ngram_tokens(prev_input_ids, num_hypos: int, no_repeat_ngram_size: int, cur_len: int, vocab_size: int):
    generated_ngrams = [{tuple([23]):[33, 31], tuple([31]):[123]} for _ in range(num_hypos)]
    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        penalty_idx = tuple(prev_input_ids[hypo_idx, cur_len - 1: cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, []) + generated_ngrams[hypo_idx].get(penalty_idx, [])

    if cur_len + 1 < no_repeat_ngram_size:
        if cur_len > 0:
            return [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    #generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            if any(e >= vocab_size for e in ngram):
                continue
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def get_model_for_inference(args, vocab_size):
    """Build the model."""

    print_rank_0('building Enc-Dec model ...')
    config = EncDecConfig.from_json_file(args.model_config)
    config.vocab_size = vocab_size
    assert not args.checkpoint_activations
    model = EncDecModelForInference(
        config,
        parallel_output=True,
        checkpoint_activations=False,
        checkpoint_num_layers=args.checkpoint_num_layers
    )

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if args.deepspeed and args.fp16:
        model.half()

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)

    # Wrap model for distributed training.
    if USE_TORCH_DDP:
        i = torch.cuda.current_device()
        model = DDP(model, device_ids=[i], output_device=i,
                    process_group=mpu.get_data_parallel_group())
    else:
        model = DDP(model)

    return model
    

def setup_model_for_inference(args, vocab_size):
    """Setup model and optimizer."""

    model = get_model_for_inference(args, vocab_size)

    if args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")

        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=None,
            args=args,
            lr_scheduler=None,
            mpu=mpu,
            dist_init_required=False
        )

    if args.load is not None:
        args.iteration = load_checkpoint(model, optimizer, lr_scheduler, args)
    else:
        args.iteration = 0

    return model


def get_masks_and_position_ids(args,
                               tokenizer,
                               contexts,
                               targets,
                               reset_position_ids,
                               reset_attention_mask):
    # Extract batch size and sequence length.
    batch_size, enc_seq_length = contexts.size()

    # Enc Attention mask.
    enc_attn_mask = torch.zeros(
        batch_size, 1, enc_seq_length, enc_seq_length, device=contexts.device)

    ctx_lengths = (contexts != tokenizer.pad_id).sum(1)
    for b in range(batch_size):
        enc_attn_mask[b, 0, :ctx_lengths[b], :ctx_lengths[b]] = 1

    # Enc Position ids.
    enc_pos_ids = torch.arange(
        enc_seq_length, dtype=torch.long, device=contexts.device)
    enc_pos_ids = enc_pos_ids.unsqueeze(0).expand_as(contexts)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        enc_pos_ids = enc_pos_ids.clone()

    batch_size, dec_seq_length = targets.size()
    # Dec Attention mask
    dec_attn_mask = torch.tril(torch.ones(
        batch_size, 1, dec_seq_length, dec_seq_length, device=targets.device))

    # Dec Position ids.
    dec_pos_ids = torch.arange(
        dec_seq_length, dtype=torch.long, device=targets.device)
    dec_pos_ids = dec_pos_ids.unsqueeze(0).expand_as(targets)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        dec_pos_ids = dec_pos_ids.clone()

    # Cross Attention Mask
    cross_attn_mask = torch.zeros(
        batch_size, 1, dec_seq_length, enc_seq_length, device=contexts.device)

    for b in range(batch_size):
        cross_attn_mask[b, 0, :, :ctx_lengths[b]] = 1

    if args.fp16:
        enc_attn_mask = enc_attn_mask.half()
        dec_attn_mask = dec_attn_mask.half()
        cross_attn_mask = cross_attn_mask.half()

    model_batch = {
        "enc_attention_mask": enc_attn_mask,
        "enc_position_ids": enc_pos_ids,
        "dec_attention_mask": dec_attn_mask,
        "dec_position_ids": dec_pos_ids,
        "cross_attention_mask": cross_attn_mask,
    }

    return model_batch


def get_inference_batch(
        context_tokens,
        device,
        batch_size,
        target_length,
        tokenizer,
        args,
    ):
    tokens = context_tokens
    tokens = tokens.view(batch_size, -1).contiguous()
    tokens = tokens.to(device)
    
    targets = torch.zeros(batch_size, target_length, dtype=torch.long, device=device) + tokenizer.get_sentinel_id(0)

    # Get the masks and postition ids.
    model_batch = get_masks_and_position_ids(
        args,
        tokenizer,
        tokens,
        targets,
        args.reset_position_ids,
        args.reset_attention_mask,
    )
    
    model_batch = {
        "enc_input_ids": tokens,
        "dec_input_ids": targets,
        **model_batch
    }

    return model_batch


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-10000, remove_unk=False):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if remove_unk:
        logits[..., 0] = filter_value

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    batch_size = logits.size()[0]
    if top_p > 0.0:
        logits=logits.view(batch_size, -1).contiguous()
        for logit in logits:
            sorted_logits, sorted_indices = torch.sort(logit, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logit[indices_to_remove] = filter_value

        logits=logits.view(batch_size, -1).contiguous()

    return logits


def generate_samples(model, tokenizer: EncDecTokenizer, args, device):
    no_repeat_ngram_size = 3
    repetition_penalty = 1.2
    batch_size = 1
    model.eval()
    with torch.no_grad():
        
        all_input_tokens = []

        while True:

            if dist.get_rank() == 0:
                input_text = input(">>> ")
                all_input_tokens.extend(tokenizer.encode(input_text) + [tokenizer.sep_id, tokenizer.get_sentinel_id(0)])
                input_len = len(all_input_tokens)
                length_tensor = torch.tensor([input_len], dtype=torch.long).to(device)
                token_tensor = torch.tensor(all_input_tokens, dtype=torch.long).to(device)
            else:
                length_tensor = torch.zeros(1, dtype=torch.long).to(device)
            
            dist.barrier()
            dist.broadcast(length_tensor, 0)

            if dist.get_rank() != 0:
                token_tensor = torch.zeros(int(length_tensor), dtype=torch.long).to(device)
            dist.broadcast(token_tensor, 0)

            token_tensor = token_tensor.unsqueeze(0)

            target_length = args.max_length

            model_batch=get_inference_batch(token_tensor, device, batch_size, target_length, tokenizer, args)

            enc_input_ids = model_batch['enc_input_ids']
            enc_attention_mask = model_batch['enc_attention_mask']
            enc_position_ids = model_batch['enc_position_ids']
            enc_outputs = model(
                enc_input_ids=enc_input_ids,
                enc_position_ids=enc_position_ids,
                enc_attention_mask=enc_attention_mask,
            )
            enc_hidden_states = enc_outputs["last_hidden_state"]
            
            # for generating responses
            # we only use the <go> token, so truncate other tokens
            dec_input_ids = model_batch['dec_input_ids'][..., :1]
            dec_attention_mask = model_batch['dec_attention_mask'][..., :1, :1]
            dec_position_ids = model_batch['dec_position_ids'][..., :1]
            # we use past_key_values, so only the current token mask is needed
            cross_attention_mask = model_batch['cross_attention_mask'][..., :1, :]
            
            unfinished_sents = enc_input_ids.new(enc_input_ids.size(0)).fill_(1)
            output_ids = enc_input_ids.new_zeros([enc_input_ids.size(0), 0])
            output_probs = torch.zeros(batch_size, 1).to(device)
            prob_idx = torch.arange(batch_size)
            past_key_values = None
            
            gen_len = 0
            # start_time = time.time()
            while gen_len < target_length:
                #print_rank_0(f'>>>>>> gen_len: {gen_len} <<<<<<')
                
                if unfinished_sents.max() == 0:
                    tokens_to_add = tokenizer.sep_id * (1 - unfinished_sents)
                    output_ids = torch.cat([output_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                
                else:
                    dec_outputs, lm_logits = model(
                        dec_input_ids=dec_input_ids,
                        dec_position_ids=dec_position_ids,
                        dec_attention_mask=dec_attention_mask,
                        cross_attention_mask=cross_attention_mask,
                        enc_hidden_states=enc_hidden_states,
                        past_key_values=past_key_values,
                    )
                    past_key_values = dec_outputs['past_key_values']
                    
                    gathered_lm_logits = [torch.zeros_like(lm_logits).to(device) for _ in range(mpu.get_model_parallel_world_size())]
                    torch.distributed.all_gather(gathered_lm_logits, lm_logits.data, mpu.get_model_parallel_group())
                    lm_logits = torch.cat(gathered_lm_logits, dim=-1)

                    logits = lm_logits[:, -1, :] / args.temperature

                    prev_output_tokens = torch.cat([enc_input_ids, output_ids], dim=-1)

                    # repetition_penalty
                    if repetition_penalty != 1.0:
                        for i in range(logits.size(0)):
                            for previous_token in set(prev_output_tokens[i].tolist()):
                                # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                                if logits[i, previous_token] < 0:
                                    logits[i, previous_token] *= repetition_penalty
                                else:
                                    logits[i, previous_token] /= repetition_penalty

                    # no_repeat_ngram_size
                    if no_repeat_ngram_size > 0:
                        banned_batch_tokens = calc_banned_ngram_tokens(
                            output_ids, logits.size(0), no_repeat_ngram_size, gen_len, logits.size(1)
                        )
                        for i, banned_tokens in enumerate(banned_batch_tokens):
                            logits[i, banned_tokens] = -1e5

                    logits = top_k_logits(logits, top_k=args.top_k, top_p=args.top_p, remove_unk=True)
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                    next_prob = probs[prob_idx, next_token]
                    tokens_to_add = next_token * unfinished_sents + tokenizer.sep_id * (1 - unfinished_sents)
                    probs_to_add = next_prob * unfinished_sents
                    output_probs = torch.cat([output_probs, probs_to_add.unsqueeze(-1)], dim=-1)
                    
                    dec_input_ids = tokens_to_add.unsqueeze(-1)
                    output_ids = torch.cat([output_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                    dec_position_ids = dec_position_ids[:, -1:] + 1
                    # let the current token attend to all previous tokens
                    dec_attention_mask = torch.cat([dec_attention_mask, dec_attention_mask[:, :, :, -1:]], dim=-1)
                    
                gen_len += 1
                unfinished_sents.mul_(tokens_to_add.ne(tokenizer.sep_id).long())

            if torch.distributed.get_rank() == 0:
                e = output_ids[0].cpu().tolist()
                generation_token_ids = e[:e.index(tokenizer.sep_id)] if tokenizer.sep_id in e else e
                all_input_tokens = all_input_tokens[:-1] + generation_token_ids + [tokenizer.sep_id]

                print(">>> {}".format(tokenizer.decode(generation_token_ids)))
                # print(tokenizer.decode(all_input_tokens))


def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    # Call the init process
    deepspeed.init_distributed()
    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mpu.model_parallel_cuda_manual_seed(seed)


def main():
    """Main serving program."""

    print('Loading Model ...')

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Arguments.
    args = get_args()

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    #get the tokenizer
    tokenizer = EncDecTokenizer(os.path.join(args.tokenizer_path, 'vocab.txt'))

    # Model, optimizer, and learning rate.
    model = setup_model_for_inference(args, tokenizer.vocab_size)
    # Timer.
    timers = Timers()

    #setting default batch size to 1
    args.batch_size = 1

    print('Model Loaded!')
    #generate samples
    generate_samples(model, tokenizer, args, torch.cuda.current_device())
    

if __name__ == "__main__":
    main()



