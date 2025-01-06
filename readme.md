## Dataset

Videos are preprocessed and converted to prompt such as this

##### Prompt That Includes the bounding boxes

`<frame> <loc_x1>364<loc_x1> <loc_y1>24<loc_y1> <loc_x2>649<loc_x2> <loc_y2>697<loc_y2> <action>demonstrating to the patient<action> <|endoftext|>
`

##### Prompt that contains all the frames for the specific action and the action
`<frame> <frame> <frame> <frame> <frame> <frame> <frame> <frame> <frame> <frame> <action>shoulder examination<action> <|endoftext|>
`

##### Prompt that takes all the actions and the final label
`<action>demonstrating to the patient<action> 1245 times <action>knee examination<action> 1139 times <action>foot examination<action> 271 times <action>pelvis check<action> 414 times <action>patient standing up<action> 48 times <action>shoulder examination<action> 699 times <action>look at the patient<action> 202 times <label>Good<label><|endoftext|>
`


## Model Card
```
MultimodalTrainer(
  (model): PhysiotherapyModel(
    (model): TrasformerForCausalLM(
      (model): TrasformerModel(
        (embed_tokens): Embedding(50264, 2048)
        (layers): ModuleList(
          (0-31): 32 x TrasformerDecoderLayer(
            (self_attn): TrasformerAttention(
              (q_proj): Linear(in_features=2048, out_features=8192, bias=False)
              (k_proj): Linear(in_features=2048, out_features=2048, bias=False)
              (v_proj): Linear(in_features=2048, out_features=2048, bias=False)
              (o_proj): Linear(in_features=8192, out_features=2048, bias=False)
              (rotary_emb): TrasformerRotaryEmbedding()
            )
            (mlp): TrasformerMLP(
              (gate_proj): Linear(in_features=2048, out_features=8192, bias=False)
              (up_proj): Linear(in_features=2048, out_features=8192, bias=False)
              (down_proj): Linear(in_features=8192, out_features=2048, bias=False)
            )
            (input_layernorm): TrasformerRMSNorm()
            (post_attention_layernorm): TrasformerRMSNorm()
          )
        )
        (norm): TrasformerRMSNorm()
      )
      (lm_head): Linear(in_features=2048, out_features=50264, bias=False)
    )
    (processor): MultimodalPreprocessor(
      (vit): ViTModel(
        (embeddings): ViTEmbeddings(
          (patch_embeddings): ViTPatchEmbeddings(
            (projection): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
          )
          (dropout): Dropout(p=0.0, inplace=False)
        )
        (encoder): ViTEncoder(
          (layer): ModuleList(
            (0-11): 12 x ViTLayer(
              (attention): ViTSdpaAttention(
                (attention): ViTSdpaSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.0, inplace=False)
                )
                (output): ViTSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.0, inplace=False)
                )
              )
              (intermediate): ViTIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): ViTOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
              )
              (layernorm_before): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (layernorm_after): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            )
          )
        )
        (layernorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        (pooler): ViTPooler(
          (dense): Linear(in_features=768, out_features=768, bias=True)
          (activation): Tanh()
        )
      )
      (model): TrasformerForCausalLM(
        (model): TrasformerModel(
          (embed_tokens): Embedding(50264, 2048)
          (layers): ModuleList(
            (0-31): 32 x TrasformerDecoderLayer(
              (self_attn): TrasformerAttention(
                (q_proj): Linear(in_features=2048, out_features=8192, bias=False)
                (k_proj): Linear(in_features=2048, out_features=2048, bias=False)
                (v_proj): Linear(in_features=2048, out_features=2048, bias=False)
                (o_proj): Linear(in_features=8192, out_features=2048, bias=False)
                (rotary_emb): TrasformerRotaryEmbedding()
              )
              (mlp): TrasformerMLP(
                (gate_proj): Linear(in_features=2048, out_features=8192, bias=False)
                (up_proj): Linear(in_features=2048, out_features=8192, bias=False)
                (down_proj): Linear(in_features=8192, out_features=2048, bias=False)
              )
              (input_layernorm): TrasformerRMSNorm()
              (post_attention_layernorm): TrasformerRMSNorm()
            )
          )
          (norm): TrasformerRMSNorm()
        )
        (lm_head): Linear(in_features=2048, out_features=50264, bias=False)
      )
      (image_projection): Linear(in_features=768, out_features=2048, bias=True)
    )
  )
  (loss): MultimodalLoss()
)
```



## Dataset
- Visit [Supervisely](https://app.supervisely.com/share-links/fbZrAAZNLtOOLvWbIm8pLWr29dm6HbI7KX22mh9iW5HuV1EH4VmfdQcvuHZKrLqi) to download videos and annotations, or you could create some and download
- Download the video path and annotations to data folder and make adjustments to the train and test csv as it suits you


## Set Up
- `pip install -r requirements.txt`
- Run `python setup.py` - Make sure to have access to kaggle hub paligemma model
- Copy the path, find `config/physionet.yaml` and replace the value of `model_id` with the path
- Then `cd ai` and `pip install -r requirements.txt`
- Also run `sudo apt update && sudo apt install ffmpeg`


## Train
- Run `python main.py`
- See logs here [WandDB Logs](https://wandb.ai/vicksemmanuel58/physionet?nw=nwuservicksemmanuel58)

## Evaluation
- Run `python eval.py`

## Inferencing
- Run `mkdir -p outputs`
- Run `python inference.py`
- Visit the localhost printed. In our case `http://127.0.0.1:7860`


## Results

![Inferencing](https://res.cloudinary.com/vickie/image/upload/v1736125606/iytnitvxicozrqzkvsgs.gif)
