# Transformer Model using LLMs


Implemented Translation Model training using the BERT pretrained model from HuggingFace.

To train the model with default parameters run `python src/translator.py train`. To evaluate the model : `python src/translator.py eval --resume_dir <resume-dir>`

For more info look at the file `src/translator.py`. For a bare-minimum implementation see : `src/en-de.py`

## Train/Eval Options

### Data Subsetting Parameters
- `maxlen`: The input data is filtered to remove sentences with token length higher than maxlen//2 for both input and output text.
- `subset`: Further filter training data randomly to `subset` number of sentences
- `subset_eval`: Filter test data randomly to `subset` number of sentences

### Model Training Parameters
- `batch_size`: batch size for model train/test
- `num_epochs`: number of epochs for training
- `scheduler_freq`: number of batches after which to `step` learning rate scheduler
- `device`: cuda-device/cpu

### Checkpoint Parameters
- `save_prefix`: save prefix for logs and model training checkpoints
- `resume_dir`: resume directory for training and evaluation
- `save_format`: checkpoint saving format

### Model Build Parameters
 Note: By default the encoder, decoder embedding and self-attention are not fine-tuned during training.

**Model Options**
- `modelpath_from`: HuggingFace model path for input language
- `modelpath_to`: HuggingFace model path for output language
- `freeze_encOnly`: if specified the encoder is frozen during training. The Decoder is fine-tuned.
- `freeze_decEmbedOnly`: if specified the encoder and the decoder Embedding is frozen during training. The rest of the decoder is fine-tuned.
- `skipScheduler`: Not Implemented
- `modelOnly_resume`: resume only the model parameters. Skip loading the optimizer and scheduler

**Optimizer & Scheduler Parameters**
- `init_lr`: initial learning rate
- `gamma`: gamma for scheduler [range: 0.1-0.5]
- `patience`: scheduler option
