import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers import CLIPVisionModel, CLIPProcessor, CLIPVisionConfig, default_data_collator
from transformers.modeling_outputs import CausalLMOutput, CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss

class PrefixedLLM(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # Project visual features to LLM dimension
        self.vision_text_proj = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)

    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.decoder.set_input_embeddings(value)

    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, labels=None, past_key_values=None, **kwargs):
        # 1. Fallback: Recover input_ids from labels if missing
        if input_ids is None:
            if labels is None:
                raise ValueError("Forward pass requires either 'input_ids' or 'labels'.")
            
            # Clone labels to create input_ids
            input_ids = labels.clone()
            
            # Sanitize: Replace -100 (ignore_index) with a valid token ID.
            # Note: We use the pad_token_id. If not defined, default to 0.
            pad_token_id = self.decoder.config.pad_token_id if self.decoder.config.pad_token_id is not None else 0
            
            # Mask replacement
            input_ids[input_ids == -100] = pad_token_id

        # 2. Check if we are in "generation mode" (cached state)
        if past_key_values is not None:
            # The prefix and previous tokens are already cached.
            # Embed only the current text tokens.
            inputs_embeds = self.decoder.get_input_embeddings()(input_ids)
            
            # In generation mode, the mask is usually updated internally or passed as 1s.
            # We use the provided attention_mask directly.
            expanded_mask = attention_mask 
        else:
            # Step 0 (First pass) or Training: Encode Image + Text

            # Encode Images
            vision_output = self.encoder(pixel_values)
            image_embeds = self.vision_text_proj(vision_output.last_hidden_state)
            
            # Embed Text
            text_embeds = self.decoder.get_input_embeddings()(input_ids)
            
            # Concatenate: [Prefix; Text]
            inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
            
            # Expand Attention Mask
            if attention_mask is not None:
                batch_size = attention_mask.shape[0]
                prefix_length = image_embeds.shape[1]
                
                # Create prefix mask (Batch, Num_Patches)
                prefix_mask = torch.ones((batch_size, prefix_length), device=attention_mask.device)
                
                # Concatenate: [1...1; Text_Mask]
                expanded_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            else:
                expanded_mask = None

        # 3. Forward pass through Decoder
        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=expanded_mask,
            past_key_values=past_key_values,
            return_dict=True,
            **kwargs 
        )
        
        logits = outputs.logits

        # 7. Calculate Loss
        loss = None
        if labels is not None:
            # We must handle the alignment between logits and labels.
            # The 'logits' include the image prefix, but 'labels' do not.
            # We usually ignore the loss on the image prefix prediction.
            
            # Shift logits: Prediction at t matches label at t+1
            # We slice off the last logit.
            shift_logits = logits[..., :-1, :].contiguous()
            
            # Prepare labels:
            # 1. Prepend -100 (ignore_index) for the visual prefix length
            prefix_length = image_embeds.shape[1]
            batch_size = labels.shape[0]
            ignore_labels = torch.full((batch_size, prefix_length), -100, device=labels.device, dtype=labels.dtype)
            
            # 2. Concatenate with text labels
            full_labels = torch.cat([ignore_labels, labels], dim=1)
            
            # 3. Shift labels: target at t+1
            shift_labels = full_labels[..., 1:].contiguous()

            # Flatten and compute CrossEntropy
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values, # Field exists in CausalLMOutputWithPast
        )

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
    
    # Ensure pixel_values are on the correct device
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
