from transformers import CLIPProcessor, CLIPModel
from transformers import CLIPTokenizer, CLIPTextModel

# CLIP model
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# text encoder
model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
# model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
# tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

def get_clip_txt_embeddings(txt):
    """
    Parameter:
        txt: list of text to be processed.
    Return:
        embeddings: numpy embeddings of given texts.
    """
    inputs = tokenizer(txt, padding=True, return_tensors="pt", max_length=77, truncation=True)
    try:
        outputs = model(**inputs)
    except:
        # the default max_position_length is 77, can't be longer than this
        print(inputs, '-------------------')
        print('inputs shape', inputs['input_ids'].shape)
        print('attention shape', inputs['attention_mask'].shape)
        assert(1==2)
        
    last_hidden_state = outputs.last_hidden_state
    pooled_output = outputs.pooler_output  # pooled (EOS token) states
    return pooled_output.detach().numpy()

if __name__ == '__main__':
    text = [" ".join(['a' for i in range(80)])]
    get_clip_txt_embeddings(text)