# mdetr-hf
This is an unofficial port of MDETR into the 🤗 Trainer ecosystem.

### To-Do
- [ ] Make sure original model can be loaded.
- [ ] Remove 2 forward calls.
- [ ] Make sure scheduler is compatible.
- [ ] Put readMe from original.
- [ ] train.py
- [ ] Allow other types of schedulers, other than linear.
- train_utils.py
    - [x] Callbacks.
    - [x] Optimizer.
- [ ] requirements.txt
- [ ] Enable Exponential Moving Average.
    - Needs custom LR-Scheduler?
    - Needs custom Optimizer.