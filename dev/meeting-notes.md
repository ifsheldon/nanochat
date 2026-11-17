## Nov 17

Everything in the branch `rlm`

1. Evaluation - ongoing:
    - SFT checkpoint of the original nanochat: just train it from scratch again
        - compare ours (effectively trained with 4K context) with 2K-context
        - on ChatCORE
        - done this week, let adi know the results
2. New Profile: 511 window with 4 cycles
    - running now
    - after pretraining + SFT, I will also run evaluation, on ChatCORE
3. Fixed a few data loading bugs -> new pretraining and SFT:
    - data loader in pretraining will discard a lot of long documents
    - similar bug in data load of SFT: long conversation discarded because of small context window not covering assistant tokens for supervision
      - fixed by increasing the cycle number to cover long context [mixed SFT]
    - whether the nanochat original loaders are fixed?
4. LoRA training for existing model:
    - Adi thinking about it
    - two potential problems or questions:
        - what dataset for finetuning? what is suitable? what's not in its original training corpus?
        - do we need to change the format of Q and A? like [context, question, answer] -> [question, context, answer] because the model does not know what is the purpose of compression when reading context first. but maybe if we train it well, it can just do good memory compression for later questions?
5. train it to write text memos along with the compressed memory [rough ideas, not TODO]
    - probably we need to do some SFT first and some RL later? but not sure about the task and rewards for now.
    - like humans does, with memory in brain and memos in hand

Cycle setting:
- Pretraining: 4 cycles, because we cannot afford more compute demands
- SFT: 4, 12, 24 cycles [mixed SFT]

TODO:
- pretraining + SFT for original nanochat, as the baseline
- pretraining + SFT for 511-4cycle profile and 1023-4cycle profile

