import torch
import torch.nn as nn
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import BaseModelOutput
from transformers import CLIPVisionConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.configuration_utils import GenerationConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from torch.nn import CrossEntropyLoss

class PrefixedLLMConfig:
    def __init__(self, encoder, decoder):
        self.encoder = encoder.config
        self.decoder = decoder.config

class PrefixedLLM(PreTrainedModel, GenerationMixin):
    def __init__(self, encoder, decoder):
        #super().__init__(PretrainedConfig())
        super().__init__(decoder.config) # avoids bugs with generate
        self.encoder = encoder
        self.config.encoder = encoder.config
        self.decoder = decoder
        self.config.decoder = decoder.config
        # self.config = PrefixedLLMConfig(encoder, decoder)
        self.generation_config = GenerationConfig()
        # Project visual features to LLM dimension
        self.vision_text_proj = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)

    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.decoder.set_input_embeddings(value)

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None: # may be passed during generation
            if pixel_values is None and input_ids is None:
                # raise error if neither pixel_values nor input_ids are provided
                # this can happen if generate() fails to pass pixel_values on first step
                raise ValueError("You have to specify either input_ids, pixel_values, or inputs_embeds")
            
            # encode image
            vision_output = self.encoder(pixel_values)  # run image through the encoder
            image_embeds = self.vision_text_proj(vision_output.last_hidden_state)  # project to LLM hidden size

            # embed text and concatenate
            if input_ids is not None:  # because generate may call the forward without input_ids
                text_embeds = self.decoder.get_input_embeddings()(input_ids)  # embed text tokens
                inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)  # concatenate image prefix with text embeddings
            else:
                inputs_embeds = image_embeds  # only image embeddings if no input_ids

        # attention mask
        if attention_mask is not None:  # works during training
            # correct attention mask (add image prefix)
            if input_ids is not None:
                prefix_mask = torch.ones((attention_mask.shape[0], image_embeds.shape[1]), device=attention_mask.device)  # create mask for image prefix
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # concatenate with text mask
            # if input_ids is None, we assume attention_mask already includes prefix or we are generating
        else:  # may be the case during generation
            attention_mask = torch.ones(inputs_embeds.shape[:-1], device=inputs_embeds.device)  # create full mask for all embeddings

        # forward through decoder
        outputs = self.decoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)  # pass embeddings through LLM

        if input_ids is not None:  # slice logits for training
            logits = outputs.logits[:, image_embeds.shape[1]:]  # ignore logits for image prefix
        else:
            logits = outputs.logits  # use all logits during generation

        return CausalLMOutputWithPast(
            loss=None,  # no loss computation in forward, handled externally
            logits=logits,  # output logits
            hidden_states=outputs.hidden_states,  # pass hidden states
            attentions=outputs.attentions,  # pass attention
            past_key_values=outputs.past_key_values  # pass past key values for caching
        )

    # # this is called by model.generate() for each generation step
    # # constructs inputs_embeds from the image prefix + input_ids
    # def prepare_inputs_for_generation(self, input_ids=None, past=None, attention_mask=None, pixel_values=None, **kwargs):
    #     # If inputs_embeds is already provided (e.g., from previous generation step), use it directly
    #     if inputs_embeds is not None:
    #         # already provided, just use it along with attention_mask and past_key_values
    #         return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask, "past_key_values": past}

    #     # encode image if pixel_values are provided
    #     if pixel_values is not None:
    #         vision_output = self.encoder(pixel_values)  # run image through encoder
    #         image_embeds = self.vision_text_proj(vision_output.last_hidden_state)  # project image features to LLM hidden size
    #     else:
    #         image_embeds = None  # no image prefix

    #     # embed text if input_ids are provided
    #     if input_ids is not None:
    #         text_embeds = self.decoder.get_input_embeddings()(input_ids)  # embed text tokens
    #         if image_embeds is not None:
    #             inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)  # concatenate image prefix + text
    #         else:
    #             inputs_embeds = text_embeds  # only text embeddings
    #     else:
    #         if image_embeds is not None:
    #             inputs_embeds = image_embeds  # only image prefix if no text
    #         else:
    #             # nothing to provide, raise error
    #             raise ValueError("You have to specify either input_ids, pixel_values, or inputs_embeds")

    #     # attention mask
    #     if attention_mask is not None:  # works during training
    #         if input_ids is not None and image_embeds is not None:
    #             # correct attention mask (add image prefix)
    #             prefix_mask = torch.ones((attention_mask.shape[0], inputs_embeds.shape[1] - input_ids.shape[1]), device=attention_mask.device)
    #             attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # concatenate image mask + text mask
    #         # if input_ids is None, assume attention_mask already includes prefix or we are generating
    #     else:  # may be the case during generation
    #         attention_mask = torch.ones(inputs_embeds.shape[:-1], device=inputs_embeds.device)  # create full mask for all embeddings

    #     # return inputs in the format expected by forward()
    #     return {
    #         "inputs_embeds": inputs_embeds,  # combined embeddings
    #         "attention_mask": attention_mask,  # attention mask
    #         "past_key_values": past  # cached past_key_values for faster generation
    #     }

# shift input ids one token to the right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):

    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

def init_decoder_input_ids(eos_token_id, bos_token_id, batch_size):
    eos = torch.tensor([eos_token_id] * batch_size).unsqueeze(-1)
    bos = torch.tensor([bos_token_id] * batch_size).unsqueeze(-1)
    decoder_input_ids = torch.cat([eos, bos], dim = 1)
    decoder_input_ids = decoder_input_ids.expand(batch_size, 2)
    return decoder_input_ids

def generate(model, tokenizer, pixel_values, max_new_tokens=20, device="cuda"):
    model.eval()
    
    batch_size = pixel_values.shape[0]

    # 1. Initialize Empty Input
    # We start with an empty text sequence: shape (Batch_Size, 0)
    # The model will therefore generate the first token purely based on the image prefix.
    input_ids = torch.zeros((batch_size, 0), dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, 0), dtype=torch.long, device=device)
    
    pixel_values = pixel_values.to(device)

    # 2. Generation Loop
    past_key_values = None
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            # If past_key_values is provided, we feed only the last token.
            # If it's the VERY FIRST step (input_ids is empty), we must feed the pixel_values.
            
            # Logic:
            # Step 0: input_ids is (B,0). We pass pixel_values. Model sees [Image]. Output predicts T1.
            # Step 1: input_ids becomes (B,1). We pass T1. Model sees [Image, T1] (via cache). Output predicts T2.
            
            if past_key_values is None:
                # First step: The "Empty Text" pass
                outputs = model(
                    input_ids=input_ids,          # (B, 0)
                    pixel_values=pixel_values,    # (B, C, H, W)
                    attention_mask=attention_mask,# (B, 0)
                    past_key_values=None
                )
            else:
                # Subsequent steps: The "Next Token" pass
                # We only feed the single most recent token
                last_token = input_ids[:, -1:]    # (B, 1)
                
                outputs = model(
                    input_ids=last_token,
                    pixel_values=None,            # Not needed when using cache
                    attention_mask=attention_mask,# Mask history is handled internally or via cache
                    past_key_values=past_key_values
                )
            
            # Get next token prediction from the last logit
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1).unsqueeze(-1)
            
            # Update cache
            past_key_values = outputs.past_key_values
            
            # Append prediction to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Update attention mask (append 1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((batch_size, 1), device=device, dtype=torch.long)], 
                dim=1
            )

            # Optimization: Stop if ALL sequences have hit EOS (optional, but good for speed)
            if (next_token == tokenizer.eos_token_id).all():
                break
                
    return input_ids

def print_trainable_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

def print_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print('total_params', total_params)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('trainable_params', trainable_params)

# define adapter
class VisionAdapter(nn.Module):
    def __init__(self, clip_dim, decoder_dim, reduction=4):
        super(VisionAdapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(clip_dim, clip_dim // reduction, bias=False),
            nn.GELU(),
            nn.Linear(clip_dim // reduction, clip_dim, bias=False),
            nn.GELU()
        )

    def forward(self, x):
        x = self.fc(x)
        return x

# wraps clip and adapter together as an encoder
class CLIPWithAdapter(PreTrainedModel):
    config_class = CLIPVisionConfig  # the CLIP config is assigned automatically as part of the init

    def __init__(self, clip_model, adapter):
        super().__init__(clip_model.config)
        self.clip = clip_model
        self.adapter = adapter

    # necessary or it triggers an error
    def get_input_embeddings(self):
        return self.clip.get_input_embeddings()

    # vision encoders don't produce token embeddings so this is none
    def get_output_embeddings(self):
        return None

    def forward(self, pixel_values, input_ids=None, attention_mask=None, **kwargs): # input_ids and attention_mask are required for compatibility (inherited requirement from PreTrainedModel)
        # pass the pixel values through CLIP
        outputs = self.clip(pixel_values=pixel_values, **kwargs)

        # extract vision features (last_hidden_state)
        vision_feats = outputs.last_hidden_state

        # apply the vision adapter to process the features
        #vision_feats = vision_feats.mean(dim=1, keepdim=True)  # add mean pooling
        adapted_feats = self.adapter(vision_feats) # apply the adapter

        # do residual-style blending as in the CLIP-Adapter paper
        new_feats = vision_feats + adapted_feats

        # return full encoder outputs (for cross-attention compatibility)
        return BaseModelOutput(
            last_hidden_state=new_feats,  # processed image embeddings
            hidden_states=outputs.hidden_states,  # encoder's intermediate states
            attentions=outputs.attentions  # attention scores (important for cross-attention)
        )
