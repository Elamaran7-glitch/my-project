
from dotenv import find_dotenv, load_dotenv
from transformers import pipline

load_dotenv(find_dotenv())

#img2text
def img2text(url):
    image_to_text=pipline("image-to-text", model="Salesforce/blip-image-captioning-base")
     
    text=image_to_text(url)[0]['generated_text']

    print(text)
    return text
#llm



#